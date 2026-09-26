import json
from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from equations import energy, latency, memory
from results_io import RESULTS, load_measurements


FIGURES = RESULTS / "figures"
B_DENSE = np.arange(1, 257)


def plot_vs_batch(m: dict[str, np.ndarray], name: str, predict: Callable[[int | np.ndarray, int | np.ndarray], np.ndarray], ylabel: str, scale: float) -> None:
    """
    Plot metric vs batch size over the whole (S, B) grid, one panel per each image size: predicted curve + measured points

    Args:
        m: measurements from load_measurements()
        name: metric column name
        predict: fn(S, B) -> predicted metric
        ylabel: y axis label with units
        scale: multiplier from SI units to plotted units
    """
    sizes = np.unique(m["S"])
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, S in zip(axes.flat, sizes):
        ax.plot(B_DENSE, predict(S, B_DENSE) * scale, color="C0", lw=1, label="predicted")
        sel = (m["S"] == S) & ~m["oom"]
        ax.scatter(m["B"][sel], m[name][sel] * scale, color="C1", s=15, zorder=3, label="measured")
        sel = (m["S"] == S) & m["oom"]
        if sel.any():
            ax.scatter(m["B"][sel], predict(S, m["B"][sel]) * scale, color="red", marker="x", s=30, zorder=3, label="OOM")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"S = {S} px")
        ax.set_xlabel("batch size B")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes.flat[len(sizes):]:
        ax.remove()
    fig.suptitle(f"{name}: predicted vs measured")
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}_vs_batch.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """
    Plot measured vs predicted latency, memory and energy to results/figures
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    m = load_measurements()
    fitted = json.loads((RESULTS / "theta.json").read_text())
    theta, theta_energy = fitted["theta"], fitted["theta_energy"]

    predict = {
        "latency": lambda S, B: latency(S, B, theta),
        "memory": lambda S, B: memory(S, B).astype(float),
        "energy": lambda S, B: energy(S, B, theta_energy),
    }
    units = {"latency": ("ms", 1e3), "memory": ("MiB", 1 / 2**20), "energy": ("J", 1.0)}
    for name, fn in predict.items():
        label, scale = units[name]
        plot_vs_batch(m, name, fn, f"{name}, {label}", scale)


if __name__ == "__main__":
    main()
