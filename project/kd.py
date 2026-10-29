import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from datasets import disable_progress_bars, load_dataset
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


TEACHER = "jhu-clsp/ettin-encoder-400m"
STUDENT = "jhu-clsp/ettin-encoder-17m" # same tokenizer as the teacher
LARGER_STUDENTS = ["jhu-clsp/ettin-encoder-32m", "jhu-clsp/ettin-encoder-68m"]
HP = {
    "teacher": {"lr": 3e-5, "epochs": 3},
    "student": {"lr": 1e-4, "epochs": 5},
}
MIN_STEPS = 300 # small subsets get more epochs
BATCH = 32
MAX_LEN = 128 # longest SST-2 sentence is 68 tokens with the Ettin tokenizer
WEIGHT_DECAY = 0.01
WARMUP = 0.06
TEMPERATURE = 2.0
ALPHA = 0.5 # weight of the KD term, 1 - ALPHA goes to CE on gold labels
SEEDS = [0, 1, 2]
SUBSET_SEED = 0 # one random order of the train set, so p1 is inside p5, p10, p25
RESULTS = Path(__file__).parent / "results"


def run_list() -> dict[str, dict]:
    """
    All runs in execution order: the 10% table, KD with full labels, the label curve, k-center,
    then teacher seeds, KD temperature and weight, and larger students

    A run is {model, checkpoint, subset, texts, teacher, temperature, alpha, seed}:
        model: "teacher" or "student", selects the hyperparameters in HP
        subset: which train examples have gold labels, "p10" is a random 10%, "kc10" is k-center 10%
        texts: "subset" trains on the labeled examples only, "all" also on the rest without gold labels
        teacher: run whose logits the student distills, None for CE on gold labels only
    """
    runs = {}

    def teacher(subset, seed=0):
        name = f"teacher_{subset}" + (f"_s{seed}" if seed else "")
        runs[name] = {"model": "teacher", "checkpoint": TEACHER, "subset": subset, "texts": "subset", "teacher": None, "temperature": TEMPERATURE, "alpha": 0.0, "seed": seed}

    def students(name, subset, texts, teacher, checkpoint=STUDENT, temperature=TEMPERATURE, alpha=ALPHA):
        for s in SEEDS:
            runs[f"{name}_s{s}"] = {
                "model": "student",
                "checkpoint": checkpoint,
                "subset": subset,
                "texts": texts,
                "teacher": teacher,
                "temperature": temperature,
                "alpha": alpha if teacher else 0.0,
                "seed": s,
            }

    teacher("p10")
    students("hard_p10", "p10", "subset", None)
    students("hard_p100", "p100", "subset", None)
    students("kd_p10", "p10", "subset", "teacher_p10")
    students("kdu_p10", "p10", "all", "teacher_p10")

    teacher("p100")
    students("kd_p100", "p100", "subset", "teacher_p100")

    for p in [1, 5, 25]:
        teacher(f"p{p}")
        students(f"hard_p{p}", f"p{p}", "subset", None)
        students(f"kdu_p{p}", f"p{p}", "all", f"teacher_p{p}")

    teacher("kc10")
    students("hard_kc10", "kc10", "subset", None)
    students("kdu_kc10", "kc10", "all", "teacher_kc10")

    for s in [1, 2]:
        teacher("p10", seed=s)
    for t in [1, 4]:
        students(f"kdu_p10_t{t}", "p10", "all", "teacher_p10", temperature=float(t))
    students("kdu_p10_a1", "p10", "all", "teacher_p10", alpha=1.0)

    for checkpoint in LARGER_STUDENTS:
        size = checkpoint.rsplit("-", 1)[1]
        students(f"hard_p10_{size}", "p10", "subset", None, checkpoint)
        students(f"kdu_p10_{size}", "p10", "all", "teacher_p10", checkpoint)
        students(f"hard_p100_{size}", "p100", "subset", None, checkpoint)
    return runs


def load_sst2(tok, limit: int | None) -> dict:
    """
    GLUE SST-2 token ids, train is the pool subsets are drawn from, validation is the test set (GLUE test labels are hidden)
    """
    ds = load_dataset("nyu-mll/glue", "sst2")
    data = {}
    for split, name in [("train", "pool"), ("validation", "test")]:
        rows = ds[split] if limit is None else ds[split].select(range(min(limit, len(ds[split]))))
        data[name] = tok(list(rows["sentence"]), truncation=True, max_length=MAX_LEN)["input_ids"]
        data[name + "_labels"] = np.array(list(rows["label"]))
    data["pad"] = tok.pad_token_id
    return data


def subset(name: str, data: dict) -> np.ndarray:
    if name.startswith("kc"):
        return np.load(RESULTS / "subsets" / f"{name}.npy")
    n = len(data["pool"])
    order = np.random.default_rng(SUBSET_SEED).permutation(n)
    return np.sort(order[:round(int(name[1:]) / 100 * n)])


def batches(ids: list, idx: np.ndarray, pad: int, device, rng=None):
    order = idx if rng is None else rng.permutation(idx)
    for i in range(0, len(order), BATCH):
        b = order[i:i + BATCH]
        x = torch.full((len(b), max(len(ids[j]) for j in b)), pad)
        for row, j in enumerate(b):
            x[row, :len(ids[j])] = torch.tensor(ids[j])
        yield b, x.to(device), (x != pad).long().to(device)


@torch.no_grad()
def predict(model, ids: list, pad: int, device) -> torch.Tensor:
    model.eval()
    out = []
    for _, x, m in batches(ids, np.arange(len(ids)), pad, device):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(input_ids=x, attention_mask=m).logits.float().cpu())
    return torch.cat(out)


def metrics(logits: torch.Tensor, labels: np.ndarray) -> dict:
    pred = logits.argmax(-1).numpy()
    tp = ((pred == 1) & (labels == 1)).sum()
    return {"acc": float((pred == labels).mean()), "f1": float(2 * tp / ((pred == 1).sum() + (labels == 1).sum()))}


def train(run: dict, data: dict, device) -> tuple[torch.nn.Module, dict]:
    """
    Fine-tune a classifier, loss = (1 - alpha) * CE on gold labels + alpha * T^2 * KL(teacher_T || student_T)

    CE is averaged over the labeled examples of a batch, KL over all of it. Without a teacher the loss is CE only.
    The last checkpoint is evaluated, there is no dev set and no early stopping.
    """
    torch.manual_seed(run["seed"])
    rng = np.random.default_rng(run["seed"])
    labeled = subset(run["subset"], data)
    idx = labeled if run["texts"] == "subset" else np.arange(len(data["pool"]))
    gold = torch.full((len(data["pool"]),), -1)
    gold[labeled] = torch.from_numpy(data["pool_labels"][labeled])
    soft = None if run["teacher"] is None else torch.load(RESULTS / "logits" / f"{run['teacher']}.pt")["pool"]
    alpha, temp = run["alpha"], run["temperature"]

    hp = HP[run["model"]]
    model = AutoModelForSequenceClassification.from_pretrained(run["checkpoint"], num_labels=2).to(device)
    steps = max(hp["epochs"] * math.ceil(len(idx) / BATCH), MIN_STEPS)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=WEIGHT_DECAY)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    step, start = 0, time.time()
    model.train()
    while step < steps:
        for b, x, m in batches(data["pool"], idx, data["pad"], device, rng):
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(input_ids=x, attention_mask=m).logits.float()
            y = gold[b].to(device)
            has = y >= 0
            ce = F.cross_entropy(logits[has], y[has]) if has.any() else logits.new_zeros(())
            loss = (1 - alpha) * ce
            if soft is not None:
                t = F.log_softmax(soft[b].float().to(device) / temp, -1)
                s = F.log_softmax(logits / temp, -1)
                loss = loss + alpha * temp**2 * F.kl_div(s, t, log_target=True, reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step == steps:
                break

    info = {
        "steps": steps,
        "n_labeled": len(labeled),
        "n_texts": len(idx),
        "labeled_positive": float(data["pool_labels"][labeled].mean()),
        "train_time": time.time() - start,
    }
    return model, info


def kcenter(name: str, data: dict, device) -> None:
    """
    K-center greedy (Sener & Savarese, 2018) on mean-pooled embeddings of the pretrained teacher, cosine distance
    """
    model = AutoModel.from_pretrained(TEACHER).to(device).eval()
    emb = []
    with torch.no_grad():
        for _, x, m in batches(data["pool"], np.arange(len(data["pool"])), data["pad"], device):
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                h = model(input_ids=x, attention_mask=m).last_hidden_state.float()
            emb.append(F.normalize((h * m[..., None]).sum(1) / m.sum(1, keepdim=True), dim=-1))
    emb = torch.cat(emb)
    k = round(int(name[2:]) / 100 * len(emb))
    first = int(np.random.default_rng(SUBSET_SEED).integers(len(emb)))
    sel = [first]
    dist = 1 - emb @ emb[first]
    for _ in range(k - 1):
        i = int(dist.argmax())
        sel.append(i)
        dist = torch.minimum(dist, 1 - emb @ emb[i])
    (RESULTS / "subsets").mkdir(parents=True, exist_ok=True)
    np.save(RESULTS / "subsets" / f"{name}.npy", np.sort(np.array(sel)))


def bench(checkpoints: list[str], data: dict, device) -> dict:
    """
    Parameters, CPU latency of one batch of 32 test sentences (median of 10), GPU throughput on the test set
    """
    out = {}
    for name in checkpoints:
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).eval()
        emb = model.get_input_embeddings().weight.numel()
        row = {"model": name, "params": sum(p.numel() for p in model.parameters()), "embedding_params": emb}
        _, x, m = next(batches(data["test"], np.arange(BATCH), data["pad"], torch.device("cpu")))
        times = []
        with torch.no_grad():
            for i in range(12):
                t = time.perf_counter()
                model(input_ids=x, attention_mask=m)
                if i >= 2:
                    times.append(time.perf_counter() - t)
        row["cpu_batch32_s"] = float(np.median(times))
        if device.type == "cuda":
            model.to(device)
            predict(model, data["test"], data["pad"], device)
            torch.cuda.synchronize()
            t = time.perf_counter()
            predict(model, data["test"], data["pad"], device)
            torch.cuda.synchronize()
            row["gpu_sentences_per_s"] = len(data["test"]) / (time.perf_counter() - t)
        out[name] = row
    out["cpu_threads"] = torch.get_num_threads()
    return out


def env(device) -> dict:
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name() if device.type == "cuda" else platform.processor(),
    }


def main() -> None:
    global RESULTS, SEEDS, MIN_STEPS, TEACHER
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", help="run names, all runs of run_list() if empty")
    parser.add_argument("--max-hours", type=float, default=11.0, help="no run starts after this time")
    parser.add_argument("--smoke", action="store_true", help="2000 train sentences, 17M model as teacher, results_smoke/")
    args = parser.parse_args()

    limit = None
    if args.smoke:
        TEACHER, limit, SEEDS, MIN_STEPS = STUDENT, 2000, [0], 20
        HP["teacher"]["epochs"] = HP["student"]["epochs"] = 1
        RESULTS = RESULTS.with_name("results_smoke")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_sst2(AutoTokenizer.from_pretrained(STUDENT), limit)
    runs = run_list()
    benches = {"bench": [TEACHER, STUDENT], "bench_sizes": [TEACHER, STUDENT, *LARGER_STUDENTS]}
    todo = args.runs or list(runs) + list(benches)
    (RESULTS / "runs").mkdir(parents=True, exist_ok=True)
    (RESULTS / "logits").mkdir(parents=True, exist_ok=True)

    start = time.time()
    for name in todo:
        path = RESULTS / "runs" / f"{name}.json"
        if path.exists():
            print(f"{name}: done before")
            continue
        if (time.time() - start) / 3600 > args.max_hours:
            print(f"stopped before {name}, {args.max_hours} h passed")
            break
        t = time.time()
        if name in benches:
            result = bench(benches[name], data, device)
        else:
            run = runs[name]
            if run["teacher"] and not (RESULTS / "logits" / f"{run['teacher']}.pt").exists():
                raise SystemExit(f"{name} distills {run['teacher']}, run it first")
            if run["subset"].startswith("kc") and not (RESULTS / "subsets" / f"{run['subset']}.npy").exists():
                kcenter(run["subset"], data, device)
            model, info = train(run, data, device)
            test = predict(model, data["test"], data["pad"], device)
            result = {**run, **info, **metrics(test, data["test_labels"])}
            if run["model"] == "teacher":
                pool = predict(model, data["pool"], data["pad"], device)
                result["pool_acc"] = metrics(pool, data["pool_labels"])["acc"]
                torch.save({"pool": pool.half(), "test": test.half()}, RESULTS / "logits" / f"{name}.pt")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        result["env"] = env(device)
        path.write_text(json.dumps(result, indent=1))
        print(f"{name}: {', '.join(f'{k} {result[k]:.4f}' for k in ['acc', 'f1'] if k in result)} in {time.time() - t:.0f} s, total {(time.time() - start) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # load reports list the new classifier head and the dropped MLM decoder for every run
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    disable_progress_bars()
    main()
