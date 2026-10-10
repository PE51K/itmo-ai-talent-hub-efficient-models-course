import csv
import json
import re
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


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
MATLAB = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E"] # default line colors of the paper's MATLAB figures
# Look of the paper's figures: Times, bold title and axis labels, boxed axes and legend, ticks inside
PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.loc": "lower right",
    "legend.fancybox": False,
    "legend.framealpha": 1,
    "legend.edgecolor": "black",
    "legend.borderaxespad": 0.3,
    "legend.handlelength": 1.5,
}
PAPER_TICKS = FormatStrFormatter("%g") # 0, 0.5, 1 as MATLAB, not 0.0, 0.5, 1.0


def load_run(name: str) -> tuple[dict, dict[str, np.ndarray]]:
    """
    Load config and metrics (one row per test) of a run written by train.py

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


def lr_axis(cfg: dict, iteration: np.ndarray) -> np.ndarray:
    """
    Calc LR of LR range test at given iterations, LR grows linearly from base_lr to max_lr over stepsize

    Args:
        cfg: run config
        iteration: iterations
    """
    return cfg["base_lr"] + (cfg["max_lr"] - cfg["base_lr"]) * iteration / cfg["stepsize"]


def plot_fig1a(runs: dict[str, tuple[dict, dict[str, np.ndarray]]]) -> None:
    """
    Plot test accuracy vs iteration of PC-LR and CLR on 50,000 samples as the paper's Fig. 1a, final accuracies in boxes

    Args:
        runs: config and metrics from load_run()
    """
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for name, label, color, text_at in [("pc_lr", "LR=0.35", MATLAB[0], (7.4e4, 0.75)), ("clr", "CLR=0.1-3", MATLAB[1], (1.9e4, 0.905))]:
        if name not in runs:
            continue
        m = runs[name][1]
        ax.plot(m["iteration"], m["test_acc"], color=color, lw=1.5, label=label)
        ax.annotate(f"{m['test_acc'][-1]:.1%}", xy=(m["iteration"][-1], m["test_acc"][-1]), xytext=text_at, ha="center", va="center",
                    bbox={"boxstyle": "square,pad=0.4", "fc": "white", "ec": "black"},
                    arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.5, "shrinkA": 0, "shrinkB": 0})
    ax.set(xlim=(0, 8e4), ylim=(0.1, 1), yticks=np.arange(0.1, 1.01, 0.1), title="Cifar10, Resnet-56", xlabel="Iteration", ylabel="Test Accuracy")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(4, 4), useMathText=True)
    ax.yaxis.set_major_formatter(PAPER_TICKS)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1a.png", dpi=150)
    plt.close(fig)


def plot_fig2b(cfg: dict, m: dict[str, np.ndarray]) -> None:
    """
    Plot test accuracy vs LR of the LR range test as the paper's Fig. 2b

    Args:
        cfg: run config
        m: run metrics from load_run()
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(lr_axis(cfg, m["iteration"]), m["test_acc"], color=MATLAB[0], lw=1.5, label=f"Max Iter={cfg['max_iter'] // 1000}k")
    ax.set(xlim=(0, 3), ylim=(0.2, 0.9), title="Cifar10, Resnet-56; LR range", xlabel="Learning Rate", ylabel="Test Accuracy")
    ax.xaxis.set_major_formatter(PAPER_TICKS)
    ax.yaxis.set_major_formatter(PAPER_TICKS)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2b.png", dpi=150)
    plt.close(fig)


def plot_fig4a(runs: dict[str, tuple[dict, dict[str, np.ndarray]]]) -> None:
    """
    Plot test accuracy vs iteration of PC-LR and CLR on 10,000 samples as the paper's Fig. 4a (Table 1 runs)

    Args:
        runs: config and metrics from load_run()
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, label, color in [("pc_lr_10k", "PC-LR; 10k samples", MATLAB[0]), ("clr_10k", "CLR; 10k samples", MATLAB[2])]:
        if name in runs:
            m = runs[name][1]
            ax.plot(m["iteration"], m["test_acc"], color=color, lw=1.5, label=label)
    ax.set(xlim=(0, 8e4), ylim=(0.1, 0.9), title="Cifar10, Resnet-56; Reduced Training set", xlabel="Iteration", ylabel="Test Accuracy")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(4, 4), useMathText=True)
    ax.yaxis.set_major_formatter(PAPER_TICKS)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4a.png", dpi=150)
    plt.close(fig)


def plot_authors_logs(runs: dict[str, tuple[dict, dict[str, np.ndarray]]], logs: dict[str, dict[str, np.ndarray]]) -> None:
    """
    Plot test accuracy of every run next to the authors' Caffe log of the same run, one panel per run

    Args:
        runs: config and metrics from load_run()
        logs: authors' logs from load_paper_log()
    """
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4), squeeze=False)
    for ax, (name, (cfg, m)) in zip(axes.flat, runs.items()):
        log = logs[name]
        if name == "lr_range_test":
            ax.plot(lr_axis(cfg, m["iteration"]), m["test_acc"], lw=1, label="ours")
            ax.plot(lr_axis(cfg, log["iteration"]), log["test_acc"], color="gray", lw=1, label="authors' log")
            ax.set_xlabel("learning rate")
        else:
            ax.plot(m["iteration"], m["test_acc"], lw=1, label=f"ours: {m['test_acc'][-1]:.1%}")
            ax.plot(log["iteration"], log["test_acc"], color="gray", lw=1, label=f"authors' log: {log['test_acc'][-1]:.1%}")
            ax.set_xlabel("iteration")
        ax.set(title=name, ylabel="test accuracy", ylim=(0, 1))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "authors_logs.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """
    Plot runs found in results/ to results/figures, print final accuracies next to the paper's and the authors' logs
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = {name: load_run(name) for name in LOG_NAMES if (RESULTS / f"{name}.csv").exists()}
    logs = {name: load_paper_log(name) for name in runs}

    plot_authors_logs(runs, logs)
    with plt.rc_context(PAPER_STYLE):
        if "lr_range_test" in runs:
            plot_fig2b(*runs["lr_range_test"])
        if "clr" in runs or "pc_lr" in runs:
            plot_fig1a(runs)
        if "clr_10k" in runs or "pc_lr_10k" in runs:
            plot_fig4a(runs)

    print(f"{'run':<14} {'train':>6} {'iterations':>12} {'paper %':>8} {'log %':>6} {'final %':>8} {'best %':>7} {'time, h':>8}")
    for name, (cfg, m) in runs.items():
        paper = f"{PAPER_ACC[name]:.1f}" if name in PAPER_ACC else "-"
        done = f"{m['iteration'][-1]:.0f}/{cfg['max_iter']}"
        print(f"{name:<14} {cfg['train_size']:>6} {done:>12} {paper:>8} {logs[name]['test_acc'][-1] * 100:>6.1f} "
              f"{m['test_acc'][-1] * 100:>8.2f} {m['test_acc'].max() * 100:>7.2f} {m['time'][-1] / 3600:>8.2f}")


if __name__ == "__main__":
    main()
