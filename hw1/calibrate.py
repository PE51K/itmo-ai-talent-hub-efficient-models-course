import json

import numpy as np
from scipy.optimize import least_squares, nnls

from equations import energy, kernels, latency, memory
from results_io import RESULTS, load_measurements


# Initial guess from Tesla T4 datasheet: ~8 TFLOP/s FP32, ~10 us per eager kernel launch
THETA_0 = {"t_launch": 1e-5, "flops_per_s": 8e12}
# Tesla T4 datasheet GDDR6 bandwidth, fixed: F_i and M_i both scale as S^2 * B, so latency data can't separate flops_per_s from bytes_per_s
BYTES_PER_S = 3.2e11


def fit_latency(S: np.ndarray, B: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    """
    Fit t_launch and flops_per_s by least squares on log(latency), so ms-scale and 100 ms-scale configs weigh the same, bytes_per_s is fixed

    Args:
        S: image sizes of fitting configs
        B: batch sizes of fitting configs
        measured: measured latencies (s)
    """
    keys = list(THETA_0)

    def residuals(log_theta: np.ndarray) -> np.ndarray:
        """
        Calc log errors of latency() for all fitting configs

        Args:
            log_theta: log of latency theta values, in THETA_0 keys order
        """
        theta = dict(zip(keys, np.exp(log_theta)), bytes_per_s=BYTES_PER_S) # log params keep theta positive and well scaled
        return np.log(latency(S, B, theta)) - np.log(measured)

    fit = least_squares(residuals, np.log([THETA_0[k] for k in keys]))
    return {**{k: float(v) for k, v in zip(keys, np.exp(fit.x))}, "bytes_per_s": BYTES_PER_S}


def fit_energy(S: np.ndarray, B: np.ndarray, measured: np.ndarray, theta: dict[str, float]) -> dict[str, float | dict[str, float]]:
    """
    Fit energy theta by non-negative least squares on relative error, energy is linear in its params once theta is fixed

    Args:
        S: image sizes of fitting configs
        B: batch sizes of fitting configs
        measured: measured energies (J)
        theta: fitted latency theta
    """
    ks = kernels(S, B)
    A = np.stack([
        sum(f for f, _ in ks), # total flops
        sum(m for _, m in ks), # total bytes moved
        latency(S, B, theta), # predicted pass duration, energy() uses the same
    ], axis=1).astype(float)
    coef, _ = nnls(A / measured[:, None], np.ones_like(measured)) # divide rows by measured -> relative error
    return {"j_per_flop": float(coef[0]), "j_per_byte": float(coef[1]), "p_static": float(coef[2]), "theta": theta}


def mape(pred: np.ndarray, measured: np.ndarray) -> float:
    """
    Calc mean absolute percentage error

    Args:
        pred: predicted values
        measured: measured values
    """
    return float(np.mean(np.abs(pred - measured) / measured) * 100)


def main() -> None:
    """
    Fit theta on base grid, write results/theta.json, report errors on fit and validation configs
    """
    m = load_measurements()
    fit = ~m["oom"] & ~m["is_validation"]
    val = ~m["oom"] & m["is_validation"]

    theta = fit_latency(m["S"][fit], m["B"][fit], m["latency"][fit])
    theta_energy = fit_energy(m["S"][fit], m["B"][fit], m["energy"][fit], theta)
    (RESULTS / "theta.json").write_text(json.dumps({"theta": theta, "theta_energy": theta_energy}, indent=2))
    print(json.dumps({"theta": theta, "theta_energy": theta_energy}, indent=2))

    pred = {
        "latency": latency(m["S"], m["B"], theta),
        "energy": energy(m["S"], m["B"], theta_energy),
        "memory": memory(m["S"], m["B"]).astype(float),
    }
    print(f"{'metric':<8} {'MAPE fit %':>11} {'MAPE val %':>11}")
    for name in pred:
        print(f"{name:<8} {mape(pred[name][fit], m[name][fit]):>11.2f} {mape(pred[name][val], m[name][val]):>11.2f}")

    # OOM boundary: memory() above total VRAM should coincide with measured OOM
    total = json.loads((RESULTS / "env.json").read_text())["gpu_total_memory"]
    predicted_oom = pred["memory"] > total
    print(f"OOM: measured {m['oom'].sum()}, predicted by memory() > {total / 2**30:.2f} GiB {predicted_oom.sum()}, "
          f"agree on {(predicted_oom == m['oom']).sum()}/{len(m['oom'])} configs")
    for S, B, p in zip(m["S"][m["oom"]], m["B"][m["oom"]], pred["memory"][m["oom"]]):
        print(f"  OOM at S={S}, B={B}: memory() = {p / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
