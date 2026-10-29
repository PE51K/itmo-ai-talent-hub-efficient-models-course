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
    ("hard_kc10", "Student, CE, k-center 10%"),
    ("kdu_kc10", "Student, KD on all texts, k-center 10%"),
    ("hard_p100", "Student, CE, 100% labels"),
    ("kd_p100", "Student, KD, 100% labels"),
]
CURVE = [1, 5, 10, 25, 100]


def load(results: Path) -> dict[str, list[dict]]:
    """
    Group runs by config, the seed suffix _s<n> dropped
    """
    groups = {}
    for path in sorted((results / "runs").glob("*.json")):
        if path.stem != "bench":
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


def bench_table(results: Path) -> str:
    path = results / "runs" / "bench.json"
    if not path.exists():
        return ""
    b = json.loads(path.read_text())
    lines = ["| Model | Params, M | Non-embedding params, M | CPU, batch of 32, ms | T4, sentences/s |", "|---|---|---|---|---|"]
    for role in ["teacher", "student"]:
        r = b[role]
        gpu = f"{r['gpu_sentences_per_s']:.0f}" if "gpu_sentences_per_s" in r else ""
        lines.append(f"| `{r['model']}` | {r['params'] / 1e6:.1f} | {(r['params'] - r['embedding_params']) / 1e6:.1f} | {r['cpu_batch32_s'] * 1000:.0f} | {gpu} |")
    return "\n".join(lines) + f"\n\nCPU latency with {b['cpu_threads']} PyTorch threads, GPU: {b['env']['device']}."


def curve(groups: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for prefix, label in [("teacher", "Teacher, CE"), ("hard", "Student, CE"), ("kdu", "Student, KD on all texts")]:
        xs, means, stds = [], [], []
        for p in CURVE:
            runs = groups.get(f"{prefix}_p{p}")
            if runs and not (prefix == "kdu" and p == 100):
                acc = np.array([r["acc"] for r in runs]) * 100
                xs.append(p)
                means.append(acc.mean())
                stds.append(acc.std(ddof=1) if len(acc) > 1 else 0)
        if xs:
            ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=label)
    if "kd_p100" in groups:
        ax.axhline(np.mean([r["acc"] for r in groups["kd_p100"]]) * 100, ls="--", c="gray", label="Student, KD, 100% labels")
    ax.set_xscale("log")
    ax.set_xticks(CURVE, [f"{p}%" for p in CURVE])
    ax.set_xlabel("Labeled share of the SST-2 train set")
    ax.set_ylabel("SST-2 validation accuracy, %")
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
    summary = table(groups) + "\n\n" + bench_table(args.results)
    (args.results / "summary.md").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
