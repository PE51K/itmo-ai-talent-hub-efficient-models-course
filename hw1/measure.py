import csv
import json
import platform
import random
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pynvml
import torch
import torch.nn as nn

from models import get_model


BASE_S = [32, 64, 128, 224, 256, 384, 512]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SEED = 0
WARMUP = 5
MIN_RUNS, MAX_RUNS = 10, 200
LATENCY_BUDGET_S = 1.0 # time spent on latency runs per config
ENERGY_WINDOW_S = 2.0 # NVML energy counter / power sensor needs a long window to resolve one pass
RESULTS = Path(__file__).parent / "results"


def sample_grid(seed: int) -> tuple[list[int], list[int]]:
    """
    Sample random S and B on top of base grid (4 S multiples of 16 in [32, 512], 3 B non powers of 2 in [1, 256])

    Args:
        seed: random seed, fixed for reproducibility
    """
    rng = random.Random(seed)
    extra_s = rng.sample([s for s in range(32, 513, 16) if s not in BASE_S], 4)
    extra_b = rng.sample([b for b in range(1, 257) if b & (b - 1) != 0], 3)
    return sorted(extra_s), sorted(extra_b)


class EnergyMeter:
    """
    Measure energy (joules) of whole GPU via NVML: total energy counter (Volta+, e.g. T4) or power sampling fallback (e.g. P100)
    """

    def __init__(self, handle: pynvml.c_nvmlDevice_t) -> None:
        """
        Pick energy source supported by GPU

        Args:
            handle: NVML device handle
        """
        self.handle = handle
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            self.counter = True
        except pynvml.NVMLError_NotSupported:
            self.counter = False

    def measure(self, fn: Callable[[], None]) -> float:
        """
        Measure energy (joules) consumed by whole GPU while fn runs

        Args:
            fn: workload, must synchronize with GPU before return
        """
        if self.counter:
            start = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) # mJ
            fn()
            return (pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) - start) / 1e3
        # Integrate power samples (trapezoid rule) in a background thread while fn() runs
        samples, done = [], threading.Event()

        def poll() -> None:
            """
            Sample GPU power (watts) every 5 ms until fn is done
            """
            while not done.is_set():
                samples.append((time.perf_counter(), pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1e3)) # mW -> W
                time.sleep(0.005)

        thread = threading.Thread(target=poll)
        thread.start()
        fn()
        done.set()
        thread.join()
        t, p = np.array(samples).T
        return float(np.trapezoid(p, t))


def measure_config(model: nn.Sequential, S: int, B: int, meter: EnergyMeter) -> tuple[float, int, float]:
    """
    Measure median latency (s), peak allocated memory (bytes) and energy (J) of one forward pass

    Args:
        model: model from models.py on cuda in eval mode
        S: input shape (w & h, only w=h supported)
        B: batch size
        meter: GPU energy meter
    """
    x = torch.randn(B, 3, S, S, device="cuda")
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def one_pass() -> float:
        """
        Run one forward pass and return its GPU time (seconds)
        """
        start.record()
        model(x) # output isn't kept, its storage is freed right after the call
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / 1e3 # ms -> s

    with torch.inference_mode():
        for _ in range(WARMUP):
            one_pass() # cuDNN heuristics, cuBLAS handle and allocator cache are initialized here

        # Peak memory: weights + x are already allocated, so they are counted as in memory()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        one_pass()
        memory = torch.cuda.max_memory_allocated()

        n = int(np.clip(LATENCY_BUDGET_S / one_pass(), MIN_RUNS, MAX_RUNS))
        latency = float(np.median([one_pass() for _ in range(n)]))

        # Back-to-back passes without per-pass sync, so the energy window has no idle gaps
        k = max(1, int(np.ceil(ENERGY_WINDOW_S / latency)))

        def energy_block() -> None:
            """
            Run k forward passes back to back
            """
            for _ in range(k):
                model(x)
            torch.cuda.synchronize()

        energy = meter.measure(energy_block) / k
    return latency, memory, energy


def main() -> None:
    """
    Measure latency, memory and energy over (S, B) grid, write results/measurements.csv and results/env.json
    """
    torch.backends.cudnn.benchmark = False # main results: PyTorch's default heuristics choose the kernel
    torch.backends.cudnn.allow_tf32 = False # so that FP32 means FP32 on Ampere and newer GPUs
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(SEED)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    meter = EnergyMeter(handle)

    extra_s, extra_b = sample_grid(SEED)
    grid_s, grid_b = sorted(BASE_S + extra_s), sorted(BASE_B + extra_b)

    RESULTS.mkdir(exist_ok=True)
    props = torch.cuda.get_device_properties(0)
    env = {
        "gpu": props.name,
        "gpu_total_memory": props.total_memory,
        "driver": pynvml.nvmlSystemGetDriverVersion(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "energy_source": "nvml_total_energy_counter" if meter.counter else "nvml_power_sampling",
        "seed": SEED,
        "extra_s": extra_s,
        "extra_b": extra_b,
    }
    (RESULTS / "env.json").write_text(json.dumps(env, indent=2))
    print(json.dumps(env, indent=2))

    rows = []
    for S in grid_s:
        model = get_model(S, device="cuda").eval() # AvgPool kernel depends on S, so model is rebuilt per S
        for B in grid_b:
            is_validation = S in extra_s or B in extra_b # base grid is used for fitting, the rest is unseen
            try:
                latency, memory, energy = measure_config(model, S, B, meter)
                row = {"S": S, "B": B, "latency": latency, "memory": memory, "energy": energy, "is_validation": is_validation}
            except torch.cuda.OutOfMemoryError:
                row = {"S": S, "B": B, "latency": "", "memory": "OOM", "energy": "", "is_validation": is_validation}
            # Frames of the caught exception are released at this point, so cached blocks can be returned
            torch.cuda.empty_cache()
            rows.append(row)
            print(row, flush=True)
        del model
        torch.cuda.empty_cache()

    with open(RESULTS / "measurements.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["S", "B", "latency", "memory", "energy", "is_validation"])
        writer.writeheader()
        writer.writerows(rows)
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
