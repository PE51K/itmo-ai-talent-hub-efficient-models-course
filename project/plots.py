import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROWS = [
    ("teacher_p10", "Teacher, 10% labels"),
    ("teacher_p100", "Teacher, 100% labels"),
    ("hard_p10", "Student, CE, 10% labels"),
    ("kd_p10", "Student, KD on the 10%"),
    ("kdu_p10", "Student, KD on all texts, 10% labels"),
    ("kdu_p10_t1", "Student, KD on all texts, 10% labels, T = 1"),
    ("kdu_p10_t4", "Student, KD on all texts, 10% labels, T = 4"),
    ("kdu_p10_a1", "Student, KD on all texts, 10% labels, KL only"),
    ("kdu_p10_t4a1", "Student, KD on all texts, 10% labels, T = 4, KL only"),
    ("kdu_p10_ts1", "Student, KD on all texts, 10% labels, teacher seed 1"),
    ("kdu_p10_ts2", "Student, KD on all texts, 10% labels, teacher seed 2"),
    ("hard_kc10", "Student, CE, k-center 10%"),
    ("kdu_kc10", "Student, KD on all texts, k-center 10%"),
    ("hard_p100", "Student, CE, 100% labels"),
    ("kd_p100", "Student, KD, 100% labels"),
]
CURVE = [1, 5, 10, 25, 100]
SIZES = ["17m", "32m", "68m"]


def load(results: Path) -> dict[str, list[dict]]:
    """
    Group runs by config, the seed suffix _s<n> dropped
    """
    groups = {}
    for path in sorted((results / "runs").glob("*.json")):
        if not path.stem.startswith("bench"):
            groups.setdefault(re.sub(r"_s\d+$", "", path.stem), []).append(json.loads(path.read_text()))
    return groups


def cell(runs: list[dict], key: str) -> str:
    v = np.array([r[key] for r in runs]) * 100
    return f"{v.mean():.1f}" if len(v) == 1 else f"{v.mean():.1f} ± {v.std(ddof=1):.1f}"


def table(groups: dict) -> str:
    lines = ["| Config | Run | Accuracy, % | F1, % | Seeds |", "|---|---|---|---|---|"]
    for key, label in ROWS:
        if key in groups:
            runs = groups[key]
            lines.append(f"| {label} | `{key}` | {cell(runs, 'acc')} | {cell(runs, 'f1')} | {len(runs)} |")
    return "\n".join(lines)


def size_table(groups: dict) -> str:
    lines = ["| Student | `hard_p10`, % | `kdu_p10`, % | `hard_p100`, % |", "|---|---|---|---|"]
    for size in SIZES:
        keys = [k if size == "17m" else f"{k}_{size}" for k in ["hard_p10", "kdu_p10", "hard_p100"]]
        if any(k in groups for k in keys):
            lines.append(f"| Ettin-encoder-{size} | " + " | ".join(cell(groups[k], "acc") if k in groups else "" for k in keys) + " |")
    return "\n".join(lines)


def bench_table(results: Path) -> str:
    paths = [results / "runs" / f"{n}.json" for n in ["bench_sizes", "bench"]]
    paths = [p for p in paths if p.exists()]
    if not paths:
        return ""
    b = json.loads(paths[0].read_text())
    lines = ["| Model | Params, M | Non-embedding params, M | CPU, batch of 32, ms | T4, sentences/s |", "|---|---|---|---|---|"]
    # bench.json of the first Kaggle run is keyed by role, later ones by checkpoint
    for r in [v for v in b.values() if isinstance(v, dict) and "model" in v]:
        gpu = f"{r['gpu_sentences_per_s']:.0f}" if "gpu_sentences_per_s" in r else ""
        lines.append(f"| `{r['model']}` | {r['params'] / 1e6:.1f} | {(r['params'] - r['embedding_params']) / 1e6:.1f} | {r['cpu_batch32_s'] * 1000:.0f} | {gpu} |")
    return "\n".join(lines) + f"\n\nCPU latency with {b['cpu_threads']} PyTorch threads, GPU: {b['env']['device']}."


def curve(groups: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for prefix, label in [("teacher", "Teacher, CE"), ("hard", "Student, CE"), ("kdu", "Student, KD on all texts")]:
        xs, means, stds, single = [], [], [], []
        for p in CURVE:
            runs = groups.get(f"{prefix}_p{p}")
            if runs and not (prefix == "kdu" and p == 100):
                acc = np.array([r["acc"] for r in runs]) * 100
                xs.append(p)
                means.append(acc.mean())
                stds.append(acc.std(ddof=1) if len(acc) > 1 else 0)
                single.append(len(acc) == 1)
        if not xs:
            continue
        line = ax.errorbar(xs, means, yerr=stds, marker="o", ms=4, capsize=4, elinewidth=1.5, label=label)
        # one-seed points have no spread to show
        ax.plot([x for x, s in zip(xs, single) if s], [m for m, s in zip(means, single) if s], "o", ms=6, mfc="white", c=line[0].get_color(), zorder=4)
    if "kd_p100" in groups:
        ax.axhline(np.mean([r["acc"] for r in groups["kd_p100"]]) * 100, ls="--", c="gray", label="Student, KD, 100% labels")
    ax.plot([], [], "o", ms=6, mfc="white", c="gray", label="1 seed, no error bar")
    ax.set_xscale("log")
    ax.set_xticks(CURVE, [f"{p}%" for p in CURVE])
    ax.set_xlabel("Labeled share of the SST-2 train set")
    ax.set_ylabel("SST-2 validation accuracy, %")
    ax.set_title("Mean ± std over 3 seeds", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    groups = load(args.results)
    (args.results / "figures").mkdir(exist_ok=True)
    curve(groups, args.results / "figures" / "label_curve.png")
    summary = table(groups) + "\n\n" + size_table(groups) + "\n\n" + bench_table(args.results)
    (args.results / "summary.md").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
