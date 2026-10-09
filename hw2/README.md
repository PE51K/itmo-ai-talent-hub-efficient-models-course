# HW2: reproducing super-convergence

Paper: Leslie N. Smith, Nicholay Topin, [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120), arXiv:1708.07120v3. The [LaTeX source](https://arxiv.org/e-print/1708.07120) unpacks into `paper/`, which is gitignored because the arXiv license doesn't allow redistribution. The official repo [lnsmith54/super-convergence](https://github.com/lnsmith54/super-convergence) has the Caffe solver and network prototxt files, `x.sh` with the changes for every figure, and the authors' training logs in `Results/`.

The [task](https://github.com/On-Point-RND/Efficient-Models-course-ITMO-2026/blob/main/home_work_two.md) is due 2026-10-10 and needs three things in this folder:

- slides explaining the paper
- a report on reproducing its results
- the code for the reproduction

## Experiments

ResNet-56 on CIFAR-10. Terms from the paper:

- Iteration: one SGD update on a batch of 1000 images (8 GPUs x 125 in the paper). Run lengths and LR schedules are counted in iterations, 50 iterations are one epoch of 50,000 images.
- LR range test (Section 3): LR starts at 0 and grows linearly during a run. Test accuracy vs LR shows which learning rates the network can still train with.
- CLR, cyclical learning rate (Section 3): LR goes linearly from a minimum to a maximum bound over `stepsize` iterations and back over the next `stepsize`, which makes one cycle. Caffe policy `triangular`.
- Super-convergence (Section 1): one CLR cycle up to a very large LR reaches higher accuracy in far fewer iterations than typical training.
- PC-LR, piecewise constant learning rate (Section 2, "typical" training): LR 0.35, divided by 10 at iterations 50,000 and 70,000. Caffe policy `multistep`.
- BN MAF (Table 1): Caffe's `moving_average_fraction`, the share of the old BN running mean/variance kept at each iteration. Running stats are what BN uses at test time.

| Run | Paper | LR policy | Training samples | Iterations | BN MAF | Paper result |
|---|---|---|---|---|---|---|
| `lr_range_test` | Fig. 2b, "Max Iter=5k" | LR 0 -> 3 linearly | 50,000 | 5,000 | 0.95 | accuracy vs LR curve |
| `clr` | Fig. 1a | CLR 0.1-3, stepsize 5,000 | 50,000 | 10,000 | 0.95 | 92.4 % |
| `pc_lr` | Fig. 1a | PC-LR 0.35 | 50,000 | 80,000 | 0.999 | 91.2 % |
| `clr_10k` | Table 1 | CLR 0.1-3, stepsize 5,000 | 10,000 | 10,000 | 0.95 | 80.6 % |
| `pc_lr_10k` | Table 1 | PC-LR 0.35 | 10,000 | 80,000 | 0.999 | 71.4 % |

## Code

[`models.py`](models.py) is the paper's Caffe ResNet-56, [`train.py`](train.py) trains the runs above with Lightning and writes `results/<run>.csv` (one row per test) and `results/<run>.json` (config and environment), [`plots.py`](plots.py) writes `results/figures/` and prints final accuracies next to the paper's and the authors' logs.

The code follows the official prototxt files and the Caffe version from the paper (October 2016, multi-GPU via P2PSync):

- Network: `Resnet56Cifar.prototxt`. Stem conv has stride 1 (Table 5 says 2), downsampling shortcuts are 3x3/2 average pooling plus zero channels, msra init for convs, xavier for the FC layer. Conv biases are left out: BN right after a conv cancels its bias, and with `lr_mult` 2, `decay_mult` 0 and init 0 it stays 0.
- Batch: Caffe normalizes BN per GPU (125 images) and averages gradients over the 8 GPUs. The root GPU runs the test net and broadcasts its BN running stats every iteration, so they come from its 125 images only. `GhostBatchNorm2d` does the same on one GPU.
- BN running stats: Caffe keeps MAF-weighted sums of batch stats and divides by the sum of weights. PyTorch's moving average starts from mean 0 / variance 1, and `GhostBatchNorm2d` removes that start at test time to get Caffe's values.
- Data: Caffe's Data layer reads the LMDB in the same order every epoch, without shuffling, and deals images to the GPUs round-robin, so GPU g gets images g, g + 8, ... of each 1000. The only augmentation is a random horizontal flip (`mirror: true`, crop size equals image size). Input is raw 0-255 pixels minus the mean image of all 50,000 training images, the 10k runs use the same `mean.binaryproto`.
- Solver: Caffe's SGD puts LR inside the momentum history (`history = 0.9 * history + lr * (grad + 1e-4 * w)`, `w -= history`), PyTorch's `SGD` doesn't, so `CaffeSGD` implements Caffe's version. Weight decay applies to everything, BN scale/shift included, and the FC bias has `lr_mult` 2. `caffe_lr` is Caffe's `triangular` and `multistep` policies; it matches every `lr =` line of the authors' logs.
- FP32 with TF32 off.
- Test: every 100 iterations, 200 batches x 125 = 25,000 images per test. That is more than the 10,000 test images, so Caffe wraps around and continues from where the previous test stopped, and `TestBatches` does the same. Caffe also tests at iteration 0, which is skipped here, but the images it reads are skipped too.

Not specified in the paper: which 10,000 images the 10k LMDB holds (the first 10,000 here) and random seeds.

The logs in `Results/` of the official repo are the authors' later replication, `plots.py` overlays them on our curves. Their PC-LR log ends at 90.3 % (91.2 % in the paper). Their 10k PC-LR log has MAF 0.9 in one BN layer (`L2_b6_cbr1_bn`), which looks like a prototxt typo, so 0.999 is used everywhere here as the paper says.

## Reproduce

Open [`hw2_colab.ipynb`](https://colab.research.google.com/github/PE51K/itmo-ai-talent-hub-efficient-models-course/blob/main/hw2/hw2_colab.ipynb) in Colab with a T4 runtime. Each run is a separate cell and they can go in separate sessions. The PC-LR runs take about 11 hours each, the others 40 min to 1.5 h.

Or open [`hw2_kaggle.ipynb`](https://kaggle.com/kernels/welcome?src=https://github.com/PE51K/itmo-ai-talent-hub-efficient-models-course/blob/main/hw2/hw2_kaggle.ipynb) in Kaggle with GPU T4 x2 and internet on. It trains two runs at once, one per GPU, in two saved versions that run in the background: the PC-LR pair (about 11 h) and the other three (about 2.5 h). `--max-hours 11.5` keeps each version inside Kaggle's 12-hour session limit.

Locally:

```
uv run python train.py lr_range_test clr
uv run python plots.py
```
