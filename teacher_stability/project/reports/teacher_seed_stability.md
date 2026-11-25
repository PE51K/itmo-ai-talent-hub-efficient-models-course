# Teacher fine-tuning seed stability on SST-2 — interim results

**Status (25 Nov 2026):** random subset experiments are complete (3 seeds each); k-center additional seeds 1 and 2 are pending. Replace this interim report by running `uv run python report_teacher_stability.py` after both runs finish.

## Scope

This contribution repeats **teacher (Ettin-encoder-400m)** cross-entropy fine-tuning with multiple random training seeds for fixed SST-2 labeled subsets; it does not replace or claim the team's knowledge-distillation experiments.

## Results from completed random-subset experiments

| Labeled share | Accuracy, % (mean ± sample std) | F1, % (mean ± sample std) | Seeds |
|---|---:|---:|---:|
| 1% | 89.64 ± 1.80 | 89.61 ± 1.74 | 3 |
| 5% | 93.39 ± 0.63 | 93.58 ± 0.58 | 3 |
| 10% | 94.07 ± 0.66 | 94.23 ± 0.63 | 3 |
| 25% | 95.18 ± 0.41 | 95.27 ± 0.40 | 3 |
| 100% | 95.83 ± 0.29 | 95.91 ± 0.26 | 3 |

## Preliminary observation

Across the random labeled subsets, the sample standard deviation of Teacher accuracy decreases from 1.80 percentage points at 1% labels to 0.29 points at 100%. The average Teacher accuracy at 25% labels is 95.18%, compared with 95.83% at 100%. The dataset subset is fixed and only the training seed varies.

## Pending: k-center 10%

The original seed-0 `teacher_kc10` run achieves **92.78% accuracy** on SST-2 validation. This single result is not a stability estimate. Run `teacher_kc10_s1` and `teacher_kc10_s2`, then generate the final report (the report generator reads the actual result JSONs and calculates all statistics).

## Limitations

- `±` is sample standard deviation from three seeds, not a confidence interval.
- Run-0 results were obtained on a Tesla T4 with an older PyTorch/Transformers environment; new runs were obtained on an RTX 4080 with a newer environment.
- Fixed labeled subsets mean this evaluates sensitivity to training initialization/shuffling rather than the randomness of dataset selection.
- The metrics are from the **SST-2 validation** split, not the hidden GLUE test set.

## Reproduction

```powershell
uv run python kd.py teacher_kc10_s1 teacher_kc10_s2
uv run python run_missing_teacher_seeds.py --include-kcenter
uv run python report_teacher_stability.py --strict
```
