# HW2: reproducing super-convergence

Paper: Leslie N. Smith, Nicholay Topin, [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120), arXiv:1708.07120v3. The [LaTeX source](https://arxiv.org/e-print/1708.07120) unpacks into `paper/`, which is gitignored because the arXiv license doesn't allow redistribution. The official repo [lnsmith54/super-convergence](https://github.com/lnsmith54/super-convergence) has only Caffe solver and network prototxt files.

The [task](https://github.com/On-Point-RND/Efficient-Models-course-ITMO-2026/blob/main/home_work_two.md) is due 2026-10-10 and needs three things in this folder:

- slides explaining the paper
- a report on reproducing its results
- the code for the reproduction

## Code

[`models.py`](models.py) is the paper's Caffe ResNet-56, [`train.py`](train.py) trains the ResNet-56 runs from the paper with Lightning and writes `results/<run>.csv`, [`plots.py`](plots.py) writes `results/figures/`.

The paper splits each 1000-image batch 8 × 125 across GPUs, and Caffe computes BN statistics per GPU. `GhostBatchNorm2d` does the same on one GPU: batch statistics per 125-image chunk, running statistics from the first chunk only, as on GPU 0 that runs the test net.

## Reproduce

Open [`hw2_colab.ipynb`](https://colab.research.google.com/github/PE51K/itmo-ai-talent-hub-efficient-models-course/blob/main/hw2/hw2_colab.ipynb) in Colab with a T4 runtime and run all cells. Locally:

```
uv run python train.py range clr pc --scale 0.2
uv run python plots.py
```

## Deviations from the paper

- FP16 mixed precision, the paper trains in FP32.
- `--scale` shortens runs (iterations and LR milestones), LR values and BN `moving_average_fraction` stay as in the paper.
- PyTorch SGD multiplies the whole velocity by the current LR, Caffe adds lr · gradient into it. The two differ only while LR changes.
