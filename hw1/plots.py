import json
from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from equations import energy, kernels, latency, memory
from results_io import RESULTS, load_measurements


FIGURES = RESULTS / "figures"
B_DENSE = np.arange(1, 257)


def plot_vs_batch(m: dict[str, np.ndarray], name: str, predict: Callable[[int | np.ndarray, int | np.ndarray], np.ndarray], ylabel: str, scale: float, total_memory: float | None = None) -> None:
    """
    Plot metric vs batch size: predicted curve + measured points per each image size

    Args:
        m: measurements from load_measurements()
        name: metric column name
        predict: fn(S, B) -> predicted metric
        ylabel: y axis label with units
        scale: multiplier from SI units to plotted units
        total_memory: GPU VRAM (bytes), drawn as OOM line on memory plot
    """
    sizes = np.unique(m["S"])
    colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))
    fig, ax = plt.subplots(figsize=(9, 6))
    for S, c in zip(sizes, colors):
        ax.plot(B_DENSE, predict(S, B_DENSE) * scale, color=c, lw=1, label=f"S={S}")
        at = m["S"] == S
        for val, face in ((False, c), (True, "none")):
            sel = at & (m["is_validation"] == val) & ~m["oom"]
            ax.scatter(m["B"][sel], m[name][sel] * scale, edgecolors=c, facecolors=face, s=25, zorder=3)
        if name == "memory":
            sel = at & m["oom"]
            ax.scatter(m["B"][sel], predict(S, m["B"][sel]) * scale, color=c, marker="x", s=40, zorder=3)
    if total_memory is not None:
        ax.axhline(total_memory * scale, color="red", ls="--", lw=1, label="GPU total memory")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size B")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{name}: lines = prediction, filled = fit, hollow = validation" + (", x = OOM" if name == "memory" else ""))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}_vs_batch.png", dpi=150)
    plt.close(fig)


def plot_parity(m: dict[str, np.ndarray], pred: dict[str, np.ndarray], units: dict[str, tuple[str, float]]) -> None:
    """
    Plot predicted vs measured for each metric, y = x is perfect prediction

    Args:
        m: measurements from load_measurements()
        pred: predicted metric per each measured config
        units: (label, scale) per each metric
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, name in zip(axes, pred):
        label, scale = units[name]
        for val, face in ((False, "C0"), (True, "none")):
            sel = (m["is_validation"] == val) & ~m["oom"]
            ax.scatter(m[name][sel] * scale, pred[name][sel] * scale, edgecolors="C0", facecolors=face, s=20,
                       label="validation" if val else "fit")
        lim = np.nanmin(m[name]) * scale / 2, np.nanmax(m[name]) * scale * 2
        ax.plot(lim, lim, "k--", lw=1, label="y = x")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"measured {name}, {label}")
        ax.set_ylabel(f"predicted {name}, {label}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "parity.png", dpi=150)
    plt.close(fig)


def plot_regimes(m: dict[str, np.ndarray], theta: dict[str, float]) -> None:
    """
    Plot which term of latency() dominates over the (S, B) plane: launch, memory or compute

    Args:
        m: measurements from load_measurements()
        theta: fitted latency theta
    """
    S, B = np.meshgrid(np.arange(32, 513, 16), B_DENSE)
    launch = np.zeros(S.shape)
    mem = np.zeros(S.shape)
    comp = np.zeros(S.shape)
    for f, b in kernels(S, B):
        t_comp, t_mem = f / theta["flops_per_s"], b / theta["bytes_per_s"]
        launch = launch + theta["t_launch"]
        mem = mem + np.where(t_mem >= t_comp, t_mem, 0)
        comp = comp + np.where(t_mem < t_comp, t_comp, 0)
    dominant = np.argmax(np.stack([launch, mem, comp]), axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = ListedColormap(["#9ecae1", "#fdae6b", "#a1d99b"])
    ax.pcolormesh(S, B, dominant, cmap=cmap, vmin=0, vmax=2, shading="nearest")
    ax.scatter(m["S"], m["B"], c="k", s=8, label="measured configs")
    for i, label in enumerate(["launch-bound", "memory-bound", "compute-bound"]):
        ax.scatter([], [], color=cmap(i), marker="s", s=80, label=label)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("image size S, px")
    ax.set_ylabel("batch size B")
    ax.set_title("dominant term of predicted latency")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "regimes.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """
    Plot measured vs predicted latency, memory, energy and latency regimes to results/figures
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    m = load_measurements()
    fitted = json.loads((RESULTS / "theta.json").read_text())
    theta, theta_energy = fitted["theta"], fitted["theta_energy"]
    total_memory = json.loads((RESULTS / "env.json").read_text())["gpu_total_memory"]

    predict = {
        "latency": lambda S, B: latency(S, B, theta),
        "memory": lambda S, B: memory(S, B).astype(float),
        "energy": lambda S, B: energy(S, B, theta_energy),
    }
    units = {"latency": ("ms", 1e3), "memory": ("MiB", 1 / 2**20), "energy": ("J", 1.0)}
    for name, fn in predict.items():
        label, scale = units[name]
        plot_vs_batch(m, name, fn, f"{name}, {label}", scale, total_memory if name == "memory" else None)
    plot_parity(m, {name: fn(m["S"], m["B"]) for name, fn in predict.items()}, units)
    plot_regimes(m, theta)


if __name__ == "__main__":
    main()
