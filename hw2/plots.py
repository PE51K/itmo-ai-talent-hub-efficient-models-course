import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS = Path(__file__).parent / "results"
FIGURES = RESULTS / "figures"
POLICY_COLORS = {"clr": "C0", "pc": "C1"}


def load_run(name: str) -> dict[str, np.ndarray]:
    """
    Load results/<name>.csv written by train.py as columns

    Args:
        name: run name from train.RUNS
    """
    with open(RESULTS / f"{name}.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def plot_range_test(name: str, cfg: dict, m: dict[str, np.ndarray]) -> None:
    """
    Plot accuracy vs LR of LR range test, analog of paper's Fig. 2b

    Args:
        name: run name
        cfg: run config as trained
        m: run metrics from load_run()
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(m["lr"], m["test_acc"] * 100, color="C0", label="test")
    ax.plot(m["lr"], m["train_acc"] * 100, color="C1", lw=1, alpha=0.7, label="train (since previous point)")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("accuracy, %")
    ax.set_title(f"LR range test: 0 -> {cfg['max_lr']} in {cfg['steps']} iterations")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}_test.png", dpi=150)
    plt.close(fig)


def plot_policies(runs: dict[str, dict], data: dict[str, dict[str, np.ndarray]]) -> None:
    """
    Plot test accuracy and LR vs iteration, one column per training set size, analog of paper's Fig. 1a and Table 1

    Args:
        runs: run configs as trained, without range tests
        data: run metrics from load_run()
    """
    sizes = sorted({cfg["train_size"] for cfg in runs.values()}, reverse=True)
    fig, axes = plt.subplots(2, len(sizes), figsize=(7 * len(sizes), 7), sharex="col", squeeze=False, height_ratios=[3, 1])
    for (ax_acc, ax_lr), size in zip(axes.T, sizes):
        for name, cfg in runs.items():
            if cfg["train_size"] != size:
                continue
            m = data[name]
            color = POLICY_COLORS[cfg["policy"]]
            ax_acc.plot(m["step"], m["test_acc"] * 100, color=color, label=f"{name}: final {m['test_acc'][-1] * 100:.1f} %")
            ax_lr.plot(m["step"], m["lr"], color=color, label=name)
        ax_acc.set_title(f"{size} training images")
        ax_acc.set_ylabel("test accuracy, %")
        ax_acc.grid(True, alpha=0.3)
        ax_acc.legend()
        ax_lr.set_yscale("log")
        ax_lr.set_xlabel("iteration")
        ax_lr.set_ylabel("learning rate")
        ax_lr.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "clr_vs_pc.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """
    Plot finished runs from results/runs.json to results/figures, print final accuracies
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = json.loads((RESULTS / "runs.json").read_text())
    data = {name: load_run(name) for name in runs}

    for name, cfg in runs.items():
        if cfg["policy"] == "range":
            plot_range_test(name, cfg, data[name])
    policies = {name: cfg for name, cfg in runs.items() if cfg["policy"] != "range"}
    if policies:
        plot_policies(policies, data)

    print(f"{'run':<8} {'train':>6} {'iters':>6} {'epochs':>7} {'final %':>8} {'best %':>7} {'time, min':>10}")
    for name, cfg in runs.items():
        m = data[name]
        print(f"{name:<8} {cfg['train_size']:>6} {cfg['steps']:>6} {m['epoch'][-1]:>7.0f} {m['test_acc'][-1] * 100:>8.2f} "
              f"{m['test_acc'].max() * 100:>7.2f} {m['time'][-1] / 60:>10.1f}")


if __name__ == "__main__":
    main()
