#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

SHARES = ("p1", "p5", "p10", "p25", "p100")
OPTIONAL = ("kc10",)
SEEDS = (0, 1, 2)


def run_name(subset: str, seed: int) -> str:
    return f"teacher_{subset}" + (f"_s{seed}" if seed else "")


def valid_result(path: Path, subset: str, seed: int) -> dict | None:
    if not path.exists():
        return None
    try:
        r = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"WARNING: can't read {path}: {e}", file=sys.stderr)
        return None
    # Avoid treating a wrong/empty result as a finished training run.
    if r.get("model") != "teacher" or r.get("subset") != subset or int(r.get("seed", -1)) != seed:
        print(f"WARNING: wrong metadata in {path}", file=sys.stderr)
        return None
    if not (0 <= r.get("acc", -1) <= 1) or not (0 <= r.get("f1", -1) <= 1):
        print(f"WARNING: missing/invalid metrics in {path}", file=sys.stderr)
        return None
    return r


def existing_and_missing(results: Path, subsets: tuple[str, ...]) -> tuple[dict, list[str]]:
    collected: dict[str, list[dict]] = {}
    missing: list[str] = []
    for subset in subsets:
        for seed in SEEDS:
            name = run_name(subset, seed)
            record = valid_result(results / "runs" / f"{name}.json", subset, seed)
            if record is None:
                missing.append(name)
            else:
                collected.setdefault(subset, []).append(record)
    return collected, missing


def write_summary(results: Path, subsets: tuple[str, ...], output: Path) -> None:
    collected, missing = existing_and_missing(results, subsets)
    rows: list[dict] = []
    for subset in subsets:
        runs = collected.get(subset, [])
        acc = [r["acc"] * 100 for r in runs]
        f1 = [r["f1"] * 100 for r in runs]
        rows.append({
            "subset": subset,
            "n_seeds": len(runs),
            "seeds": ",".join(str(r["seed"]) for r in runs),
            "accuracy_mean_pct": f"{statistics.mean(acc):.5f}" if acc else "",
            "accuracy_sample_std_pct": f"{statistics.stdev(acc):.5f}" if len(acc) > 1 else "",
            "f1_mean_pct": f"{statistics.mean(f1):.5f}" if f1 else "",
            "f1_sample_std_pct": f"{statistics.stdev(f1):.5f}" if len(f1) > 1 else "",
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {output}")
    if missing:
        print(f"Still missing ({len(missing)}): {', '.join(missing)}")
    else:
        print("All selected teacher experiments have three seeds.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Actually train missing seeds; default is preview only")
    parser.add_argument("--include-kcenter", action="store_true", help="Also repeat teacher_kc10 with seeds 1 and 2")
    parser.add_argument("--max-hours", type=float, default=10.0, help="kd.py does not start a new run after this many hours")
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--summary", type=Path, default=None, help="CSV output path (default: results/teacher_seed_summary.csv)")
    args = parser.parse_args()
    if args.max_hours <= 0:
        parser.error("--max-hours must be positive")
    subsets = SHARES + (OPTIONAL if args.include_kcenter else ())
    results = args.results.resolve()
    collected, missing = existing_and_missing(results, subsets)
    print("Existing teacher runs:")
    for subset in subsets:
        have = sorted(r["seed"] for r in collected.get(subset, []))
        print(f"  {subset:5s}: {have}, target {list(SEEDS)}")
    print(f"Missing ({len(missing)}): {', '.join(missing) if missing else 'none'}")

    project_dir = Path(__file__).resolve().parent
    if not (project_dir / "kd.py").exists():
        parser.error("Place this script beside kd.py in project/")
    if args.execute and results != (project_dir / "results").resolve():
        parser.error("--execute uses kd.py's built-in project/results; remove --results for training")
    if args.execute and missing:
        command = [sys.executable, "kd.py", *missing, "--max-hours", str(args.max_hours)]
        print("Executing sequentially:", " ".join(command), flush=True)
        status = subprocess.run(command, cwd=project_dir, check=False).returncode
        if status != 0:
            print("Training stopped with an error; existing completed runs have been kept.", file=sys.stderr)
            write_summary(results, subsets, (args.summary or results / "teacher_seed_summary.csv").resolve())
            return status
        # Source plots.py auto-discovers _s1 and _s2 runs and redraws mean ± std.
        print("Updating plots.py outputs...")
        subprocess.run([sys.executable, "plots.py", "--results", str(results)], cwd=project_dir, check=True)
    elif missing:
        print("PREVIEW ONLY. Run with --execute to start the GPU training.")
    write_summary(results, subsets, (args.summary or results / "teacher_seed_summary.csv").resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
