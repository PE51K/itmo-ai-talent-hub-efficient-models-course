# Project plan: knowledge distillation into a tiny model on 10% of the data

Topic from the course list: "Data-Efficient Training — Knowledge Distillation от большой к tiny модели на маленьком 10% данных". Deadline and defense: 05.12.2026, 10 minutes per team, 10 points.

## What the course asks for

- Code in a repository with a README that reproduces the results. A Dockerfile with pinned dependencies or an image on Docker Hub is a plus.
- The README has everything the slides have, shorter, plus how to run and the results.
- Detailed slides: motivation, plan of work, frameworks and libraries, a review of the papers used, results, and conclusions with next steps (this part can take several slides). Slides that don't fit in 10 minutes can be left for offline reading.

## Previous iterations

ITMO 2025 had three projects on data-efficient training, links in the [2025 README](https://github.com/On-Point-RND/Efficient-Models-course-ITMO-2025). The course doesn't publish grades, so what got marks has to be read from the criteria and from what the projects show. The SHAD 2026 iteration has one NAS + KD project ([binary-wolfishness/nas-kd](https://github.com/binary-wolfishness/nas-kd/tree/nas-evolution-kd-hinton)) on NATS-Bench with ResNet students; its slides are a Git LFS file that didn't download.

| Project | Model, data | Methods | Full data | Random 10% | Best 10% |
|---|---|---|---|---|---|
| [ITMO-EDLM/data-efficiency](https://github.com/ITMO-EDLM/data-efficiency), [slides](https://docs.google.com/presentation/d/1ezU5i-dlzWrnJaix4gswlcNKfJifOUsj3TPGTdwMOnY) (our exact topic) | ModernBERT-base, SST-2 | 10 selection strategies, ClearML, HP search for the full run | 0.947 | 0.758 | 0.921 (entropy + k-center) |
| [IsachenkoBogdan/llm-data-selection-bert](https://github.com/IsachenkoBogdan/llm-data-selection-bert), [slides](https://docs.google.com/presentation/d/1WH8Rehl0b8xZYDhjvW41tIw_-lu6J7JcqxJvmUXBJgc) | bert-tiny, SST-2 | quality / diversity / importance selectors, LLM-as-judge, Hydra + Snakemake + W&B | 0.819 | 0.747 | 0.792 (quality + diversity) |
| [Bovrrr/...-train-data-sampling](https://github.com/Bovrrr/itmo-edlm-course-project-train-data-sampling), [slides](https://docs.google.com/presentation/d/1uBuXALY-gL28Q0zb4bIBhyf6BZderT-Bjen2o7VacJg) | ModernBERT-base, SST-2 | random, high/low confidence, k-means, 5% random + 5% hard | 0.951 | 0.921 | 0.929 (k-means) |

What they have in common: all three are about choosing which 10% to train on. None of them distills: the exact-topic repo has no teacher and only a CE / BCE loss (`src/data_efficiency/utils/loss.py`). So the KD part of our topic is still open, and it's what we build the project around.

What to take from them:

- The slide structure that worked: survey-based taxonomy of methods (Qin et al., [Unleashing the Power of Data Tsunami](https://arxiv.org/abs/2408.02085)), one slide per method with a scheme, a results table, conclusions, future work. The bert-tiny team also reported selection compute time next to accuracy, which fits an efficiency course.
- SST-2 with accuracy / F1 on the GLUE validation split. Using it again makes our numbers directly comparable to theirs.
- Their findings to check under KD: random is a strong baseline, diversity-based selection beats confidence-based, "easy" subsets (low perplexity, high confidence) hurt.

What to avoid:

- An undertuned random baseline. The same ModernBERT-base on random 10% gets 0.758 in one project and 0.921 in another, which turns "97% of full quality" into a 0.004 gain over random. We tune the baseline with the same budget as every other run.
- One run per configuration. On 6.7k examples a few points move between seeds; we report mean ± std over 3 seeds.
- READMEs without a run command for the results, `__pycache__` committed, results only in screenshots.

## Setup

GPU is only Colab and Kaggle (T4, 16 GB, fp16, no bf16), same as HW2. Everything has to fit in a Kaggle session (12 h) and resume from a checkpoint.

- Task: GLUE SST-2, 67k train sentences. Validation (872) is the test set, since GLUE test labels are hidden. A dev split for early stopping and HP search is carved from the training pool.
- Budget: labels for 10% of the train set (6.7k). The other 90% of the sentences are available without labels; in practice unlabeled text is cheap and labels are not. Every configuration states which of the two it uses.
- Teacher: a large encoder fine-tuned on the 10% labels only, e.g. ModernBERT-large or RoBERTa-large (335M-395M). Pick in week 1 by whichever trains stably in fp16 on a T4. A teacher trained on full data is reported separately as a reference, since it uses labels outside the budget.
- Student: `google/bert_uncased_L-2_H-128_A-2` (bert-tiny, 4.4M) as the main one, `L-4_H-256` (bert-mini, 11M) as the second size. These are the compact models from [Well-Read Students Learn Better](https://arxiv.org/abs/1908.08962).
- Metrics: accuracy and F1, plus parameters, CPU latency per batch, training and teacher inference time. Teacher-vs-student compression and speed-up go on the summary slide.

## Experiments

Ordered by priority. 1-4 are the minimum project, 5-6 are what makes it more than a rerun of 2025, 7 if time is left.

1. Baselines, student with hard labels: random 10% and 100%. Same tuning budget for both.
2. Teacher on 10%; also the full-data teacher as a reference.
3. Logit KD on the same 10% ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)): CE + KL on softened logits, sweep T and alpha.
4. KD on unlabeled data: the teacher labels the other 90% with soft targets, the student trains on 10% gold + 90% soft. This is the main hypothesis: with a 10% label budget, distillation through unlabeled text recovers most of the gap to 100%. Related: [Tang et al., 2019](https://arxiv.org/abs/1903.12136), [Noisy Student](https://arxiv.org/abs/1911.04252).
5. Selection + KD: which 10% to label, given that a teacher will be trained on them. Random, k-center ([Sener & Savarese, 2018](https://arxiv.org/abs/1708.00489)) and entropy + diversity, the best strategy from 2025. Does selection still matter once the student also sees the soft-labeled rest?
6. Intermediate-layer distillation, hidden states and attention as in [TinyBERT](https://arxiv.org/abs/1909.10351), vs logits only. Needs a projection, since the student's hidden size is 128 or 256.
7. Data-size curve: 1%, 5%, 10%, 25% labels for the best KD setup vs hard-label training.

Optional: quantize the best student to int8 on CPU (lecture 5) for one more latency row.

## Papers for the review slides

- KD: [Hinton et al., 2015](https://arxiv.org/abs/1503.02531); [DistilBERT](https://arxiv.org/abs/1910.01108); [TinyBERT](https://arxiv.org/abs/1909.10351); [MiniLM](https://arxiv.org/abs/2002.10957); [Well-Read Students Learn Better](https://arxiv.org/abs/1908.08962); [Beyer et al., 2022, a good teacher is patient and consistent](https://arxiv.org/abs/2106.05237); [Stanton et al., 2021, Does KD really work?](https://arxiv.org/abs/2106.05945).
- Unlabeled data and self-training: [Tang et al., 2019](https://arxiv.org/abs/1903.12136); [Noisy Student](https://arxiv.org/abs/1911.04252).
- Data selection: [Qin et al., survey](https://arxiv.org/abs/2408.02085); [core-set / k-center](https://arxiv.org/abs/1708.00489); [Data Diet, EL2N](https://arxiv.org/abs/2107.07075).

## Timeline

Six weeks from 24.10. The data-efficiency lecture is on 09.11.

| Week | Dates | Work |
|---|---|---|
| 1 | 26.10-01.11 | Pipeline: data split, train/eval for teacher and student, Kaggle notebook with resume. Pick the teacher. Experiments 1-2 |
| 2 | 02.11-08.11 | Experiments 3-4, T / alpha sweep |
| 3 | 09.11-15.11 | Experiment 5 |
| 4 | 16.11-22.11 | Experiment 6, 7 if time is left. Freeze the result set |
| 5 | 23.11-29.11 | Figures, README, Dockerfile, slides with the paper review |
| 6 | 30.11-05.12 | Rerun from README on a clean Kaggle session, rehearse the 10-minute talk |

## Repository

`train.py` with one config per experiment, results as CSV / JSON in `results/` and figures from `plots.py`, as in HW2. A Kaggle notebook runs the experiment list, a Colab one runs a single config. Dockerfile with pinned versions from `uv.lock`. If teammates need write access, this folder moves to a separate team repository.
