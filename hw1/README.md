# HW1: analytical performance model of a small CNN

Derivations are in `hw1_handwritten.pdf`, the equations are in `equations.py`.

## Environment

Google Colab, Tesla T4 (14.56 GiB), driver 580.82.07, CUDA 12.8, cuDNN 9.19.0, PyTorch 2.11.0+cu128, Python 3.13.15. Full record in `results/env.json`.

## Reproduce

On Colab with a T4 runtime:

```
!git clone https://github.com/PE51K/itmo-ai-talent-hub-efficient-models-course.git repo
%cd repo/hw1
!pip install -q nvidia-ml-py
!python measure.py    # results/measurements.csv, results/env.json
!python calibrate.py  # results/theta.json
!python plots.py      # results/figures/*.png
```

## Results

θ is fitted on the base grid (63 configs), validation is the 69 configs with a random S or B.

| Parameter | Value |
|---|---|
| t_launch | 25.1 µs |
| P | 3.47 TFLOP/s |
| W | 320 GB/s (T4 datasheet, fixed) |
| e_F | 0 J/FLOP |
| e_M | 0.27 nJ/byte |
| P_static | 47.6 W |

| MAPE | Fit | Validation |
|---|---|---|
| Latency | 17.6 % | 22.3 % |
| Energy | 14.9 % | 23.1 % |
| Memory | 20.8 % | 24.7 % |

No OOM on the grid, and `memory()` predicts none either.

![latency](results/figures/latency_vs_batch.png)
![memory](results/figures/memory_vs_batch.png)
![energy](results/figures/energy_vs_batch.png)

## Discussion

- Regimes: latency sits on the launch floor (17 kernels × t_launch ≈ 0.43 ms) up to B ≈ 71 at S = 32, B ≈ 18 at S = 64, B ≈ 5 at S = 128 and B ≤ 2 at S ≥ 208. Past it the pass is compute-bound: ~87 % of the time is convs, ~12 % is ReLU/pool kernels, which are always memory-bound (0.1–0.3 FLOP/byte vs P/W ≈ 11). Convs are memory-bound only inside the launch-bound corner, so no separate memory-bound stage is visible.
- Memory is underpredicted, most likely by a cuDNN conv workspace that `memory()` doesn't model: past a shape-dependent threshold (S = 128 / B = 64, S = 256 / B = 16, but not S = 512 / B = 8) peak memory jumps by 24–34 bytes per S²·B. Tiny batches get a fixed 5–10 MiB extra instead.
