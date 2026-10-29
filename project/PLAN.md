# Plan: knowledge distillation into a 17M encoder with 10% of the labels

Course topic: "Data-Efficient Training — Knowledge Distillation от большой к tiny модели на маленьком 10% данных". Submission and defense: 05.12.2026, 10 minutes per team. The course requires code with a README that reproduces the results (a Dockerfile with pinned dependencies is a plus) and detailed slides: motivation, plan of work, libraries, a review of the papers used, results, conclusions and next steps.

## Method

Knowledge distillation (KD, [Hinton et al., 2015](https://arxiv.org/abs/1503.02531)) trains a small student model to match the output distribution of a large teacher model, in addition to or in place of gold labels. The teacher's probabilities over classes carry more information per example than a one-hot label, and the teacher can label texts that have no gold label. The project measures how much of the gap between a student trained on 10% of the labels and a student trained on 100% of them KD closes.

The budget is gold labels for 10% of the SST-2 train set (6,735 of 67,349 sentences). The other 90% of the sentences are available without labels. The GLUE validation split (872 sentences) is the test set, because GLUE test labels are hidden. There is no dev set: every run uses fixed hyperparameters and is evaluated at its last step.

| Role | Model | Parameters |
|---|---|---|
| Teacher | [`jhu-clsp/ettin-encoder-400m`](https://huggingface.co/jhu-clsp/ettin-encoder-400m) | 400M |
| Student | [`jhu-clsp/ettin-encoder-17m`](https://huggingface.co/jhu-clsp/ettin-encoder-17m) | 17M, of them 12.9M in the embedding matrix |

Both are Ettin encoders ([Weller et al., 2025](https://arxiv.org/abs/2507.11412)): the ModernBERT architecture ([Warner et al., 2024](https://arxiv.org/abs/2412.13663)) trained with one recipe and one tokenizer at sizes from 17M to 1B. The 1B encoder does not fit a T4 for fine-tuning with AdamW in fp16.

The student loss is `(1 - a) * CE(student, gold) + a * T^2 * KL(teacher_T || student_T)` with `a = 0.5` and temperature `T = 2`; CE (cross-entropy) is averaged over the examples of a batch that have a gold label, KL over all of them. Without a teacher, `a = 0`.

| Config | Labeled examples | Training texts | Teacher |
|---|---|---|---|
| `teacher_p10` | random 10% | the labeled 10% | none |
| `hard_p10` | random 10% | the labeled 10% | none |
| `kd_p10` | random 10% | the labeled 10% | `teacher_p10` |
| `kdu_p10` | random 10% | all 67,349 | `teacher_p10` |
| `hard_p100` | 100% | all 67,349 | none |
| `kd_p100` | 100% | all 67,349 | `teacher_p100` |

The same configs at 1%, 5%, and 25% give the label curve; `hard_kc10` and `kdu_kc10` replace the random 10% by k-center greedy selection ([Sener & Savarese, 2018](https://arxiv.org/abs/1708.00489)) on mean-pooled embeddings of the pretrained teacher. The random subsets are nested: the 1% is inside the 5%, the 10%, and the 25%.

Hyperparameters: AdamW, weight decay 0.01, linear warmup over 6% of steps and linear decay, batch 32, fp16 autocast. Teacher: LR 3e-5, 3 epochs. Student: LR 1e-4, 5 epochs. A run takes at least 300 steps, which gives the 1% and 5% subsets more epochs. Student configs run with seeds 0, 1, 2; teacher configs with seed 0.

## Questions and the comparisons that answer them

1. How much does KD recover at a 10% label budget? Gap closed = `(kdu_p10 - hard_p10) / (hard_p100 - hard_p10)` on test accuracy.
2. Does the gain come from the teacher's soft labels on the same examples or from the unlabeled 90%? `kd_p10` vs `kdu_p10`.
3. How does the gain change with the label budget? `hard` vs `kdu` at 1%, 5%, 10%, 25%.
4. Does selecting which 10% to label still matter under KD? `hard_kc10 - hard_p10` vs `kdu_kc10 - kdu_p10`.
5. What does the student cost compared to the teacher? Parameters, CPU latency for a batch of 32, T4 throughput.

A difference counts as a result when it exceeds the sum of the two standard deviations over seeds.

## Code

| File | Content |
|---|---|
| `kd.py` | Data, training, KD loss, k-center, benchmark, the list of runs. Writes `results/runs/<run>.json`, teacher logits to `results/logits/`, k-center indices to `results/subsets/`. |
| `plots.py` | `results/summary.md` (mean ± std per config) and `results/figures/label_curve.png`. |
| `kd_kaggle.ipynb` | Runs `kd.py` on Kaggle or Colab, resumes from earlier results, zips `results/`. |

`uv run python kd.py --smoke` checks the whole pipeline on CPU in a few minutes: 2,000 train sentences, the 17M model as its own teacher, one seed, results in `results_smoke/`.

## Five days

Day 1. Push the code. On Kaggle, run the notebook interactively with `RUNS = "teacher_p10 hard_p10_s0 kdu_p10_s0"` and check that losses are finite in fp16, the teacher reaches at least 90% test accuracy, and the measured run times fit the 12-hour session. Then start the full version with Save & Run All.

Day 2. Download the output of the full version. If it stopped before `bench`, add its output as input to a new version and run it again; finished runs are skipped. Check the results: `teacher_p100` at 95% test accuracy or higher (ModernBERT-large reports 97.1% on SST-2), the seeds of one config within about 1 point.

Day 3. Answer questions 1-5 from `results/summary.md` and the label curve. Write the README report: setup, results table, figure, benchmark, how to reproduce. Rerun any config whose seeds disagree by more than 2 points with 2 more seeds.

Day 4. Slides: motivation, the method with the loss, the papers, setup, the 10% table, the label curve, selection under KD, cost of teacher vs student, conclusions, next steps. Compare with the 2025 projects on the same task (table below).

Day 5. Dockerfile with versions pinned from `uv.lock`. Rerun `teacher_p10`, `hard_p10_s0`, `kdu_p10_s0` from the README instructions in a clean Kaggle session and check the numbers match within the seed spread. Rehearse the talk to 10 minutes.

## Previous projects on SST-2 with a 10% budget

The 2025 ITMO course had three projects on training with 10% of SST-2 ([course README](https://github.com/On-Point-RND/Efficient-Models-course-ITMO-2025)). All three compared data selection strategies with cross-entropy training; none used a teacher.

| Project | Model | 100% | Random 10% | Best 10% |
|---|---|---|---|---|
| [ITMO-EDLM/data-efficiency](https://github.com/ITMO-EDLM/data-efficiency) | ModernBERT-base | 94.7 | 75.8 | 92.1, entropy + k-center |
| [IsachenkoBogdan/llm-data-selection-bert](https://github.com/IsachenkoBogdan/llm-data-selection-bert) | bert-tiny, 4.4M | 81.9 | 74.7 | 79.2, quality + diversity |
| [Bovrrr/itmo-edlm-course-project-train-data-sampling](https://github.com/Bovrrr/itmo-edlm-course-project-train-data-sampling) | ModernBERT-base | 95.1 | 92.1 | 92.9, k-means |

Accuracy in %, one run per config in each project. The two ModernBERT-base projects report random 10% at 75.8 and 92.1 for the same model and data.

## Papers for the review

- KD: [Hinton et al., 2015](https://arxiv.org/abs/1503.02531); [DistilBERT](https://arxiv.org/abs/1910.01108); [TinyBERT](https://arxiv.org/abs/1909.10351); [MiniLM](https://arxiv.org/abs/2002.10957); [Turc et al., 2019, Well-Read Students Learn Better](https://arxiv.org/abs/1908.08962); [Beyer et al., 2022, a good teacher is patient and consistent](https://arxiv.org/abs/2106.05237); [Stanton et al., 2021, Does knowledge distillation really work?](https://arxiv.org/abs/2106.05945).
- Distillation through unlabeled text: [Tang et al., 2019](https://arxiv.org/abs/1903.12136); [Noisy Student](https://arxiv.org/abs/1911.04252).
- Models: [ModernBERT](https://arxiv.org/abs/2412.13663); [Ettin](https://arxiv.org/abs/2507.11412).
- Data selection: [Sener & Savarese, 2018](https://arxiv.org/abs/1708.00489); [Qin et al., 2024, survey](https://arxiv.org/abs/2408.02085).
