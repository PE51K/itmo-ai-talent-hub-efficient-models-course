from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

SUBSETS = ("p1", "p5", "p10", "p25", "p100", "kc10")
SEEDS = (0, 1, 2)
ROOT = Path(__file__).resolve().parent


def name(subset: str, seed: int) -> str:
    return f"teacher_{subset}" + (f"_s{seed}" if seed else "")


def read_runs(results: Path) -> tuple[dict[str, list[dict]], list[str]]:
    groups: dict[str, list[dict]] = {}
    missing = []
    # seeds results building
    for subset in SUBSETS:
        group = []
        for seed in SEEDS:
            run = name(subset, seed)
            path = results / "runs" / f"{run}.json"
            if not path.is_file():
                missing.append(run)
                continue
            obj = json.loads(path.read_text(encoding="utf-8"))
            if obj.get("model") != "teacher" or obj.get("subset") != subset or obj.get("seed") != seed:
                raise ValueError(f"Unexpected metadata in {path}")
            for key in ("acc", "f1"):
                value = obj.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid {key} in {path}")
            group.append(obj)
        groups[subset] = group
    return groups, missing


def stats(rows: list[dict], metric: str) -> tuple[float, float | None]:
    xs = [100.0 * row[metric] for row in rows]
    if not xs:
        return math.nan, None
    return statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else None


def stat_cell(rows: list[dict], metric: str) -> str:
    if not rows:
        return "not available"
    mean, std = stats(rows, metric)
    return f"{mean:.2f} ± {std:.2f}" if std is not None else f"{mean:.2f} (1 seed)"


def make_figure(groups: dict[str, list[dict]], path: Path) -> bool:
    if not groups["p10"] or not groups["kc10"]:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    means, errs, labels = [], [], []
    for key, label in (("p10", "Random 10%"), ("kc10", "k-center 10%")):
        mean, std = stats(groups[key], "acc")
        means.append(mean)
        errs.append(std or 0)
        labels.append(label)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar([0, 1], means, yerr=errs, marker="o", linestyle="none", capsize=5)
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel("SST-2 validation accuracy, %")
    ax.set_title("Teacher: random vs k-center (mean ± sample std)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def make_report(groups: dict[str, list[dict]], missing: list[str], figure_rel: str | None) -> str:
    rows = [
        "# Teacher fine-tuning seed stability on SST-2",
        "",
        "## Scope of this contribution",
        "",
        "This report covers **additional teacher fine-tuning seeds only**. It does not claim credit for the team's "
        "KD student experiments, model implementations or original seed-0 runs.",
        "",
        "- Model: `jhu-clsp/ettin-encoder-400m` (teacher); objective: cross-entropy classification.",
        "- Dataset: GLUE SST-2; evaluation: its labeled validation split (872 sentences).",
        "- Training settings follow `kd.py`: AdamW, batch size 32, learning rate 3e-5, "
        "3 epochs with at least 300 steps, fp16 autocast when CUDA is available.",
        "- Random labeled-data shares: 1%, 5%, 10%, 25%, 100%; plus `kc10` (10% k-center greedy).",
        "- Seeds 0, 1, 2; **the labeled subset is held fixed** because `SUBSET_SEED = 0`. "
        "For k-center, the same saved `results/subsets/kc10.npy` is reused.",
        "- `±` means sample standard deviation across available fine-tuning seeds (`ddof=1`), "
        "not a confidence interval.",
        "",
        "## Results",
        "",
        "| Labeled selection | Accuracy, % | F1, % | Seeds |",
        "|---|---:|---:|---:|",
    ]
    display = (("p1", "Random 1%"), ("p5", "Random 5%"), ("p10", "Random 10%"),
               ("p25", "Random 25%"), ("p100", "Random 100%"), ("kc10", "k-center 10%"))
    for subset, label in display:
        runs = groups[subset]
        rows.append(f"| {label} | {stat_cell(runs, 'acc')} | {stat_cell(runs, 'f1')} | {len(runs)}/3 |")
    rows += ["", "## Interpretation", ""]
    if all(len(groups[k]) == 3 for k in ("p1", "p100")):
        _, sd1 = stats(groups["p1"], "acc")
        _, sd100 = stats(groups["p100"], "acc")
        rows.append(
            f"- The teacher accuracy spread is {sd1:.2f} percentage points at 1% labels and "
            f"{sd100:.2f} points at 100% labels; it decreases by roughly {sd1 / sd100:.1f}× "
            "in this three-seed sample."
        )
    if all(len(groups[k]) == 3 for k in ("p25", "p100")):
        a25, _ = stats(groups["p25"], "acc")
        a100, _ = stats(groups["p100"], "acc")
        rows.append(f"- Moving from 25% to 100% labels changes mean teacher accuracy by "
                    f"{a100 - a25:+.2f} percentage points in these runs.")
    if len(groups["p10"]) == 3 and len(groups["kc10"]) == 3:
        p10_by_seed = {r["seed"]: 100 * r["acc"] for r in groups["p10"]}
        kc_by_seed = {r["seed"]: 100 * r["acc"] for r in groups["kc10"]}
        diffs = [kc_by_seed[s] - p10_by_seed[s] for s in SEEDS]
        mean_diff = statistics.mean(diffs)
        sd_diff = statistics.stdev(diffs)
        rows.append(f"- k-center 10% minus random 10%: "
                    f"{mean_diff:+.2f} percentage points averaged over paired seed IDs "
                    f"(sample std of paired differences {sd_diff:.2f} points). "
                    "This is an observational comparison, not evidence of statistical significance.")
    elif groups["kc10"]:
        rows.append("- k-center has fewer than three training seeds; **do not yet make a stability "
                    "claim or a mean ± std comparison with random 10%**.")
    if figure_rel:
        rows += ["", f"![Teacher 10% selection comparison]({figure_rel})"]
    rows += [
        "", "## Limits and reproducibility", "",
        "- Three seeds give a preliminary estimate of variation, not reliable statistical significance.",
        "- Because subset selection is fixed, this probes **training-seed variability**. "
        "It does **not** test selection-seed variability or generalization to a new test set.",
        "- Run 0 in the supplied team results was evaluated on a Tesla T4 with an earlier "
        "software environment; newly added runs may use an RTX 4080 and newer PyTorch. "
        "Hardware/software differences can also affect measured variation.",
        "- These numbers are validation scores, not scores on the hidden GLUE test labels.",
        "- Reproduce from `project/` with `uv run python run_missing_teacher_seeds.py --include-kcenter --execute`; "
        "existing completed JSONs are skipped. Then run "
        "`uv run python report_teacher_stability.py` to regenerate this report and its comparison figure.",
        "", "## Run provenance", "",
        "| Run | Accuracy, % | F1, % | Device | PyTorch |",
        "|---|---:|---:|---|---|",
    ]
    for subset in SUBSETS:
        for run in groups[subset]:
            env = run.get("env") or {}
            rows.append(f"| `{name(subset, run['seed'])}` | {100 * run['acc']:.2f} | "
                        f"{100 * run['f1']:.2f} | {env.get('device', 'unknown')} | "
                        f"{env.get('torch', 'unknown')} |")
    if missing:
        rows += ["", f"**Pending runs ({len(missing)}):** " + ", ".join(f"`{n}`" for n in missing) + "."]
    else:
        rows += ["", "All six teacher configurations have results for seeds 0, 1, and 2."]
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "teacher_seed_stability.md")
    parser.add_argument("--plot", type=Path, default=ROOT / "reports" / "teacher_kcenter_vs_random.png")
    parser.add_argument("--strict", action="store_true", help="Fail if any of 18 expected results is missing")
    args = parser.parse_args()
    groups, missing = read_runs(args.results)
    if missing and args.strict:
        parser.error(f"Missing runs: {', '.join(missing)}")
    img_written = make_figure(groups, args.plot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rel = args.plot.name if img_written and args.plot.parent.resolve() == args.output.parent.resolve() else None
    args.output.write_text(make_report(groups, missing, rel), encoding="utf-8")
    print(f"Report: {args.output}")
    if img_written:
        print(f"Figure: {args.plot}")
    if missing:
        print(f"NOTE: {len(missing)} runs are missing: {', '.join(missing)}")
    else:
        print("All 18 teacher seed results are available.")


if __name__ == "__main__":
    main()
