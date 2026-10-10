# HW2: reproducing super-convergence

Paper: Leslie N. Smith, Nicholay Topin, [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120), arXiv:1708.07120v3.

- slides explaining the paper: [`slides.pdf`](slides.pdf) (source [`slides.tex`](slides.tex))
- reproduction report: this README
- code: [`models.py`](models.py) (ResNet-56), [`train.py`](train.py) (training), [`plots.py`](plots.py) (figures), [`hw2_kaggle.ipynb`](hw2_kaggle.ipynb) (the run that produced [`results/`](results))

## Results

ResNet-56 on CIFAR-10, batch 1000, one run per configuration on Kaggle's 2 x T4.

| Run | Paper | Schedule | Paper, % | Authors' log, % | Ours, % | Time, h |
|---|---|---|---|---|---|---|
| `lr_range_test` | Fig. 2b | LR 0 -> 3 over 5,000 iterations | | | | 0.46 |
| `clr` | Fig. 1a | one CLR cycle 0.1 -> 3 -> 0.1, 10,000 iterations | 92.4 | 92.4 | 91.8 | 0.92 |
| `pc_lr` | Fig. 1a | LR 0.35, x0.1 at 50k and 70k, 80,000 iterations | 91.2 | 90.3 | 91.5 | 7.40 |
| `clr_10k`, `pc_lr_10k` | Table 1 | the same on 10,000 training samples | 80.6, 71.4 | 80.6, 71.2 | not reproduced | |

"Authors' log" is the final test accuracy in the authors' Caffe logs from the [official repo](https://github.com/lnsmith54/super-convergence).

![Fig. 1a](results/figures/fig1a.png)

Super-convergence is reproduced. CLR reaches 91.8 % in 10,000 iterations, while PC-LR needs 80,000 to get 91.5 %. That is 8 times less GPU time.

![Fig. 2b](results/figures/fig2b.png)

The LR range test stays at 0.7-0.8 test accuracy up to LR 3, as in the paper. This is what tells the paper that super-convergence is possible.

![Comparison with the authors' logs](results/figures/authors_logs.png)

Our curves follow the authors' runs. CLR ends 0.6 points below the paper and PC-LR 0.3 above. The authors' own PC-LR log is 0.9 points below their paper, so differences of this size appear between runs.

Not reproduced, for lack of GPU time: Table 1 with reduced training sets (the 10,000-sample runs are implemented), Fig. 1b with other stepsizes, the LR estimate of Section 4, and other datasets and architectures.

## Setup

The code follows the authors' Caffe files and logs from the official repo:

- ResNet-56 from `Resnet56Cifar.prototxt`, FP32.
- The batch of 1000 is 8 GPUs x 125 images in the paper. `CaffeBatchNorm2d` normalizes every 125 images separately and takes the running stats from the first GPU's 125.
- `CaffeSGD` is Caffe's SGD, with LR inside momentum and weight decay 1e-4 on all parameters. `caffe_lr` gives the same LR as the authors' logs.
- No shuffling, random flips only, mean image subtracted. The model is tested every 100 iterations on 25,000 images, wrapping around the test set as Caffe does.
- `train.py` runs one process per GPU and splits every batch between them as the paper's 8 GPUs do. The environment of each run is in `results/<run>.json`.

## Reproduce

Kaggle: open [`hw2_kaggle.ipynb`](https://kaggle.com/kernels/welcome?src=https://github.com/PE51K/itmo-ai-talent-hub-efficient-models-course/blob/main/hw2/hw2_kaggle.ipynb) with GPU T4 x2 and internet on. `VERSION = 1` runs the table's 50k runs in about 9 h, `VERSION = 2` runs the 10k runs.

Colab, one T4: [`hw2_colab.ipynb`](https://colab.research.google.com/github/PE51K/itmo-ai-talent-hub-efficient-models-course/blob/main/hw2/hw2_colab.ipynb). PC-LR runs don't fit in a 12-hour session there.

Locally:

```
uv run python train.py lr_range_test clr pc_lr  # results/<run>.csv, results/<run>.json
uv run python plots.py                          # results/figures/
```
