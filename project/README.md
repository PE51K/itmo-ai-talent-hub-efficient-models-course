# Knowledge distillation into a 17M encoder with 10% of SST-2 labels

A 17M-parameter student distilled from a 400M teacher through all SST-2 train sentences reached 88.1% test accuracy with gold labels for 10% of them, against 84.3% with cross-entropy on the same labels and 89.5% with all labels. Group project of the ITMO "Efficient models" course, defense on 05.12.2026; method, questions, and schedule are in [`PLAN.md`](PLAN.md).

## Setup

The teacher is [Ettin-encoder-400m](https://huggingface.co/jhu-clsp/ettin-encoder-400m), the student [Ettin-encoder-17m](https://huggingface.co/jhu-clsp/ettin-encoder-17m): ModernBERT-architecture encoders from one pretraining recipe with one tokenizer ([Weller et al., 2025](https://arxiv.org/abs/2507.11412)). The task is GLUE SST-2 (binary sentiment, 67,349 train sentences); the GLUE validation split (872 sentences) is the test set. A random p% of the train set has gold labels; the random subsets are nested.

Knowledge distillation (KD, [Hinton et al., 2015](https://arxiv.org/abs/1503.02531)) trains the student on `0.5 * CE(student, gold) + 0.5 * T^2 * KL(teacher_T || student_T)` with temperature `T = 2`, where CE (cross-entropy) is averaged over the examples that have a gold label. Run names:

| Run | Student training data | Loss |
|---|---|---|
| `teacher_p<p>` | p% labeled sentences | CE, the 400M teacher |
| `hard_p<p>` | p% labeled sentences | CE |
| `kd_p<p>` | p% labeled sentences | CE + KL from `teacher_p<p>` |
| `kdu_p<p>` | all 67,349 sentences, p% of them labeled | CE where labeled + KL from `teacher_p<p>` everywhere |

`kc10` in place of `p10` is a 10% chosen by k-center greedy ([Sener & Savarese, 2018](https://arxiv.org/abs/1708.00489)) on mean-pooled embeddings of the pretrained teacher. Hyperparameters are fixed, with no dev set: AdamW, batch 32, fp16; teacher LR 3e-5 for 3 epochs, student LR 1e-4 for 5 epochs, at least 300 steps. Students ran with 3 seeds, teachers with 1. All 45 runs and the benchmark took 197 min on one Kaggle T4.

## KD on unlabeled sentences closes 72% of the gap at 10% labels

| Run | Test accuracy, % | F1, % |
|---|---|---|
| `teacher_p10` | 93.7 | 93.8 |
| `hard_p10` | 84.3 ± 1.3 | 84.7 ± 1.2 |
| `kd_p10` | 84.1 ± 0.6 | 84.7 ± 0.5 |
| `kdu_p10` | 88.1 ± 0.2 | 88.4 ± 0.1 |
| `hard_p100` | 89.5 ± 0.6 | 89.8 ± 0.5 |
| `kd_p100` | 89.0 ± 0.3 | 89.4 ± 0.3 |

Mean ± sample standard deviation over 3 seeds; full table in [`results/summary.md`](results/summary.md), per-run files in [`results/runs/`](results/runs).

`kdu_p10` gained 3.8 points over `hard_p10`, 72% of the 5.2-point gap to `hard_p100`. The gain came from the 90% of sentences without gold labels: KD on the labeled 10% alone (`kd_p10`) was within the seed spread of `hard_p10`. With all labels, KD from `teacher_p100` (95.9%) did not raise the student either: `kd_p100` and `hard_p100` differ by 0.5 points, less than the sum of their standard deviations. The student stayed 5.6 points below its teacher at 10% labels and 6.4 points below at 100%.

## With KD, 5% of the labels beat 25% without it

![Accuracy vs labeled share](results/figures/label_curve.png)

| Labeled share | `teacher`, % | `hard`, % | `kdu`, % |
|---|---|---|---|
| 1% (673 sentences) | 87.6 | 73.4 ± 3.4 | 84.3 ± 0.5 |
| 5% (3,367) | 94.0 | 81.5 ± 0.8 | 88.3 ± 0.3 |
| 10% (6,735) | 93.7 | 84.3 ± 1.3 | 88.1 ± 0.2 |
| 25% (16,837) | 95.1 | 86.7 ± 0.8 | 89.2 ± 0.4 |
| 100% (67,349) | 95.9 | 89.5 ± 0.6 | |

KD on all sentences with 1% of the labels matched CE with 10% (84.3% both), and with 5% it exceeded CE with 25% (88.3% vs 86.7%). The KD gain shrank from 10.9 points at 1% to 2.6 points at 25%. At 25% the KD student (89.2%) was within the seed spread of the student trained with CE on all labels (89.5%). The seed spread of the KD student was 0.2-0.5 points at every share, against 0.6-3.4 for CE. Teacher numbers are single runs: `teacher_p5` scored above `teacher_p10` by 0.3 points.

## K-center selection of the 10% lowered accuracy

| Run | Random 10%, % | K-center 10%, % |
|---|---|---|
| `teacher` | 93.7 | 92.8 |
| `hard` | 84.3 ± 1.3 | 79.1 ± 1.0 |
| `kdu` | 88.1 ± 0.2 | 86.7 ± 0.3 |

K-center selection cost the CE student 5.2 points and the KD student 1.4 points. The k-center subset had 57.3% positive labels against 55.2% in the random subset and 55.8% in the full train set.

## The student is 65 times faster than the teacher on CPU

| Model | Parameters, M | Without embeddings, M | CPU, batch of 32, ms | T4 fp16, sentences/s |
|---|---|---|---|---|
| Ettin-encoder-400m | 395.8 | 344.3 | 6,042 | 312 |
| Ettin-encoder-17m | 16.9 | 4.0 | 93 | 3,350 |

CPU latency is the median of 10 forward passes over 32 test sentences with 2 PyTorch threads on a Kaggle CPU; T4 throughput is measured on the 872 test sentences ([`results/runs/bench.json`](results/runs/bench.json)). The student has 23 times fewer parameters and 86 times fewer outside the embedding matrix.

## Limits

Hyperparameters were not tuned for any run, and `T = 2`, `a = 0.5` were not varied. Teachers ran with one seed, so differences between teachers below 1 point are not results. Whether the student's 89.5% with all labels is the limit of a 17M encoder on SST-2 or of the training recipe was not tested.

## Reproduce

Kaggle: open [`kd_kaggle.ipynb`](https://kaggle.com/kernels/welcome?src=https://github.com/PE51K/itmo-ai-talent-hub-efficient-models-course/blob/main/project/kd_kaggle.ipynb) with a GPU and internet on, then Save Version -> Save & Run All. The notebook describes how to resume a stopped version from its output.

Colab, one T4: [`kd_kaggle.ipynb`](https://colab.research.google.com/github/PE51K/itmo-ai-talent-hub-efficient-models-course/blob/main/project/kd_kaggle.ipynb).

Locally:

```
uv sync
uv run python kd.py --smoke      # whole pipeline on CPU, 2,000 sentences, results_smoke/
uv run python kd.py              # all runs, needs a CUDA GPU, results/runs/<run>.json
uv run python kd.py kdu_p10_s0   # one run
uv run python plots.py           # results/summary.md, results/figures/
```

Environment of the reported runs: Kaggle, Tesla T4, Python 3.13.15, PyTorch 2.11.0+cu128, Transformers 5.16.1.
