# HW1: analytical performance model of a small CNN

FLOPs, memory, latency and energy of the CNN from `models.py` as closed-form functions of image size S and batch size B, checked against measurements on a Tesla T4. Derivations are in `hw1_handwritten.pdf`, the code is in `equations.py`.

The spec says "After each convolution layer we use ReLU", so the stem is Conv7×7 s2 → ReLU → MaxPool 3×3 s2 p1.

## Environment

Google Colab, free tier:

- GPU: Tesla T4, 14.56 GiB, driver 580.82.07
- CUDA 12.8, cuDNN 9.19.0
- PyTorch 2.11.0+cu128, Python 3.13.15
- Energy source: NVML total energy counter

The full record is in `results/env.json`.

## Reproduce

On Colab with a T4 runtime:

```
!git clone https://github.com/PE51K/itmo-ai-talent-hub-efficient-models-course.git repo
%cd repo/hw1
!pip install -q nvidia-ml-py
!python measure.py    # ~15 min, writes results/measurements.csv and results/env.json
!python calibrate.py  # writes results/theta.json, prints MAPE and OOM check
!python plots.py      # writes results/figures/*.png
```

Locally on an NVIDIA GPU: `uv run python measure.py`, then the same two scripts. `calibrate.py` and `plots.py` only read `results/`, so they don't need a GPU.

`measure.py` sets the flags from the spec (`cudnn.benchmark = False`, TF32 off) and feeds `torch.randn` inputs. For each (S, B):

- 5 warm-up passes;
- memory: `max_memory_allocated()` over one pass after `reset_peak_memory_stats()`;
- latency: median of 10-200 passes, each timed with CUDA events;
- energy: NVML energy counter over at least 2 s of back-to-back passes, divided by the number of passes;
- `torch.cuda.OutOfMemoryError` is caught and recorded as `OOM`.

Grid: base S {32, 64, 128, 224, 256, 384, 512} × B {1, 2, 4, 8, 16, 32, 64, 128, 256}, plus random S {80, 208, 304, 320} and B {111, 133, 139} (seed 0), 132 configs in total. θ is fitted on the 63 base configs. The other 69 have a random S or B and are only used for validation (`is_validation` in `measurements.csv`).

## Results

```
FLOPs(S, B)   = B (17751 S² + 313600)
Memory(S, B)  = 12 680 976 + 68 S² B                   bytes
Bytes(S, B)   = 4 (95 S² B + 2148 B + 1 040 324)       bytes moved
Latency(S, B) = Σ over 17 kernels [t_launch + max(F_i / P, M_i / W)]
Energy(S, B)  = e_F FLOPs + e_M Bytes + P_static Latency
```

Memory is weights (4.16 MB) + cuBLAS workspace (8.52 MB) + input x + input and output of the stem MaxPool with its int64 indices, which is the largest pair in the net. Bytes moved assumes each kernel reads its inputs and weights from DRAM once and writes its output once.

Conv and Linear FLOPs match `torch.utils.flop_counter.FlopCounterMode` exactly (it doesn't count ReLU and pooling).

Fitted θ, `results/theta.json`:

| Parameter | Value |
|---|---|
| t_launch | 25.1 µs |
| P | 3.47 TFLOP/s |
| W | 320 GB/s, T4 datasheet, fixed |
| e_F | 0 J/FLOP |
| e_M | 0.27 nJ/byte |
| P_static | 47.6 W |

MAPE of predicted vs measured:

| | Fit, 63 configs | Validation, 69 configs |
|---|---|---|
| Latency | 17.6 % | 22.3 % |
| Energy | 14.9 % | 23.1 % |
| Memory | 20.8 % | 24.7 % |

No config hit OOM, and `memory()` doesn't predict one either (132/132 agree). The largest config, S=512 with B=256, peaks at 5.75 GiB measured vs 4.26 GiB predicted out of 14.56 GiB. `memory()` puts the first OOM at S=512 around B=876, outside the grid.

![latency](results/figures/latency_vs_batch.png)
![memory](results/figures/memory_vs_batch.png)
![energy](results/figures/energy_vs_batch.png)

## Discussion

### Regimes

Launch-bound. 17 kernels × 25 µs = 0.43 ms is the floor: S=32 stays at 0.47-0.49 ms from B=2 to B=8. Launch overhead is more than half of the predicted time up to S²B ≈ 6.5·10⁴ (S=32 with B=64, S=128 with B=4, S=224 with B=1).

Memory-bound. ReLU, MaxPool and AvgPool do 0.13-0.29 FLOP per byte, far below the ridge point P/W ≈ 11 FLOP/byte, so they're memory-bound at any S and B. The two Linear layers are memory-bound up to B ≈ 25-30, since they read 0.63 MB of weights for a few samples. Convs do 43-267 FLOP per byte of activations. So memory-bound kernels never take more than 12 % of the predicted time, and the whole pass goes from launch-bound straight to compute-bound. The memory-bound regime shows up per kernel, not for the pass.

Compute-bound. From S²B ≈ 7·10⁵ (S=224 with B=16) launch is under 10 % and latency grows linearly in S²B, slope 1 on the log-log plots. P = 3.47 TFLOP/s is 43 % of the T4 FP32 peak (8.1 TFLOP/s). Measured throughput at S=512 with B=64 is 3.67 TFLOP/s.

### Where the model breaks

The residuals aren't random because they come from effects the model leaves out, not from measurement noise.

Step between B=64 and B=111. At every S, latency per sample jumps 1.3-1.6× between B=64 and B=128 and stays there. Measured / predicted is 0.87 (median) for B ≤ 64 and 1.33 for B ≥ 111, and with one P the fit lands between the two levels. Energy has the same step (0.93 → 1.41), since it goes through latency. Most likely cuDNN's heuristic (`benchmark = False`) switches to another conv algorithm for large batches. I didn't profile the kernels to confirm it.

Memory gap. Measured memory is above predicted by:

- about 1 MiB for mid-size configs at every S;
- 6-10 MiB at S ≤ 128 with small B;
- a gap proportional to B above a threshold that drops as S grows (S=512 from B=16, S=224 from B=32, S=128 from B=64, S=32 and S=64 from B=111), up to 1.5 GiB at S=512 with B=256.

The last two look like cuDNN workspace, which depends on the algorithm cuDNN picks per shape. `memory()` has no calibrated parameters, so it can't follow that.

One P for all kernels. A 1×1 conv on a 2×2 map (S=32) and a 5×5 conv on a 128×128 map reach different fractions of peak, and one cuDNN conv can launch more than one kernel. This is the spread at small S: S=64 with B=16-64 is 20-30 % faster than predicted, S=32 with B=16 is 23 % slower.

S=32 with B=1 is 1.5× predicted. It's the first config measured, so the GPU is probably not fully warmed up after 5 passes.

Parameters that can't be separated. All F_i and M_i scale as S²B, so latency data fixes only the sum of F_i/P and M_i/W, not P and W apart: a free fit gives the same error for any split. That's why W is fixed at the datasheet 320 GB/s and only t_launch and P are fitted. For the same reason NNLS puts all dynamic energy on bytes and e_F = 0. P_static = 47.6 W lies between the measured average power of the smallest configs (31 W) and the largest (71 W, the T4 TDP is 70 W): in the compute-bound regime latency also scales as S²B, so P_static · Latency takes part of the dynamic energy. The split between static and dynamic energy isn't physical, only the total is checked.
