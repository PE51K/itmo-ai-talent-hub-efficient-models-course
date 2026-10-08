import csv
import json
import re
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS = Path(__file__).parent / "results"
FIGURES = RESULTS / "figures"
LOGS = Path(__file__).parent / "data" / "paper_logs"
LOG_URL = "https://raw.githubusercontent.com/lnsmith54/super-convergence/master/Results/"
# Authors' Caffe logs of the same runs, from Results/ of the official repo
LOG_NAMES = {
    "lr_range_test": "range3Iter5kFig2b",
    "clr": "clr3SS5kFig1a",
    "pc_lr": "lr35Fig1a",
    "clr_10k": "clr3SS5kTr10kTab1",
    "pc_lr_10k": "lr35Tr10kTab1",
}
PAPER_ACC = {"clr": 92.4, "pc_lr": 91.2, "clr_10k": 80.6, "pc_lr_10k": 71.4} # final test accuracy, %, Fig. 1a and Table 1
POLICY_LABELS = {"triangular": "CLR 0.1-3", "multistep": "PC-LR 0.35"}
POLICY_COLORS = {"triangular": "C0", "multistep": "C1"}


def load_run(name: str) -> tuple[dict, dict[str, np.ndarray]]:
    """
    Load config and metrics (one row per test) of a run written by train.py, the CSV may be partial if the run was stopped

    Args:
        name: run name from train.RUNS
    """
    cfg = json.loads((RESULTS / f"{name}.json").read_text())["config"]
    with open(RESULTS / f"{name}.csv") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return cfg, {k: np.array([float(r[k]) for r in rows]) for k in reader.fieldnames}


def load_paper_log(name: str) -> dict[str, np.ndarray]:
    """
    Download (once) and parse test accuracy of the authors' Caffe log of a run

    Args:
        name: run name from train.RUNS
    """
    path = LOGS / LOG_NAMES[name]
    if not path.exists():
        LOGS.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LOG_URL + LOG_NAMES[name], path)
    text = path.read_text()
    return {
        "iteration": np.array([int(m) for m in re.findall(r"Iteration (\d+), Testing net", text)]),
        "test_acc": np.array([float(m) for m in re.findall(r"Test net output #0: accuracy = ([\d.]+)", text)]),
    }


def plot_range_test(cfg: dict, m: dict[str, np.ndarray], log: dict[str, np.ndarray]) -> None:
    """
    Plot test accuracy vs LR of LR range test against the authors' log, analog of paper's Fig. 2b

    Args:
        cfg: run config as trained
        m: run metrics from load_run()
        log: authors' log from load_paper_log()
    """
    lr_per_iteration = cfg["max_lr"] / cfg["stepsize"] # LR grows linearly from base_lr 0
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(m["iteration"] * lr_per_iteration, m["test_acc"] * 100, color="C0", label="ours")
    ax.plot(log["iteration"] * lr_per_iteration, log["test_acc"] * 100, color="gray", ls="--", lw=1, label="authors' log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("test accuracy, %")
    ax.set_title(f"LR range test: 0 -> {cfg['max_lr']} in {cfg['max_iter']} iterations")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "lr_range_test.png", dpi=150)
    plt.close(fig)


def plot_policies(runs: dict[str, tuple[dict, dict[str, np.ndarray]]], logs: dict[str, dict[str, np.ndarray]]) -> None:
    """
    Plot test accuracy and LR vs iteration of CLR and PC-LR runs against the authors' logs, one column per training set size, analog of paper's Fig. 1a and Table 1

    Args:
        runs: config and metrics from load_run() of CLR and PC-LR runs
        logs: authors' logs from load_paper_log()
    """
    sizes = sorted({cfg["train_size"] for cfg, _ in runs.values()}, reverse=True)
    fig, axes = plt.subplots(2, len(sizes), figsize=(7 * len(sizes), 7), sharex="col", squeeze=False, height_ratios=[3, 1])
    for (ax_acc, ax_lr), size in zip(axes.T, sizes):
        for name, (cfg, m) in runs.items():
            if cfg["train_size"] != size:
                continue
            color, label = POLICY_COLORS[cfg["lr_policy"]], POLICY_LABELS[cfg["lr_policy"]]
            log = logs[name]
            ax_acc.plot(m["iteration"], m["test_acc"] * 100, color=color, lw=1, label=f"{label}, ours: {m['test_acc'][-1] * 100:.1f} %")
            ax_acc.plot(log["iteration"], log["test_acc"] * 100, color=color, ls="--", lw=1, alpha=0.6,
                        label=f"{label}, authors' log: {log['test_acc'][-1] * 100:.1f} %")
            ax_lr.plot(m["iteration"], m["lr"], color=color, label=label)
        ax_acc.set_title(f"{size:,} training samples")
        ax_acc.set_ylabel("test accuracy, %")
        ax_acc.grid(True, alpha=0.3)
        ax_acc.legend()
        ax_lr.set_yscale("log")
        ax_lr.set_xlabel("iteration")
        ax_lr.set_ylabel("learning rate")
        ax_lr.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "clr_vs_pc_lr.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """
    Plot runs found in results/ to results/figures, print final accuracies next to the paper's and the authors' logs
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = {}
    for name in LOG_NAMES:
        if not (RESULTS / f"{name}.csv").exists():
            continue
        cfg, m = load_run(name)
        if len(m["iteration"]): # a run stopped before its first test has only the header
            runs[name] = (cfg, m)
    logs = {name: load_paper_log(name) for name in runs}

    if "lr_range_test" in runs:
        plot_range_test(*runs["lr_range_test"], logs["lr_range_test"])
    policies = {name: run for name, run in runs.items() if name != "lr_range_test"}
    if policies:
        plot_policies(policies, logs)

    print(f"{'run':<14} {'train':>6} {'iterations':>12} {'paper %':>8} {'log %':>6} {'final %':>8} {'best %':>7} {'time, h':>8}")
    for name, (cfg, m) in runs.items():
        paper = f"{PAPER_ACC[name]:.1f}" if name in PAPER_ACC else "-"
        done = f"{m['iteration'][-1]:.0f}/{cfg['max_iter']}"
        print(f"{name:<14} {cfg['train_size']:>6} {done:>12} {paper:>8} {logs[name]['test_acc'][-1] * 100:>6.1f} "
              f"{m['test_acc'][-1] * 100:>8.2f} {m['test_acc'].max() * 100:>7.2f} {m['time'][-1] / 3600:>8.2f}")


if __name__ == "__main__":
    main()
