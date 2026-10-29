# Knowledge distillation into a 17M encoder with 10% of SST-2 labels

Group project of the ITMO "Efficient models" course, defense on 05.12.2026. A 400M Ettin encoder trained on 10% of the SST-2 labels is the teacher; a 17M Ettin encoder is the student, trained with cross-entropy on gold labels, with KD on the same 10%, or with KD on all 67,349 train sentences. Method, runs, and the schedule are in [`PLAN.md`](PLAN.md).

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
