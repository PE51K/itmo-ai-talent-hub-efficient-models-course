import csv
from pathlib import Path

import numpy as np


RESULTS = Path(__file__).parent / "results"


def load_measurements() -> dict[str, np.ndarray]:
    """
    Load results/measurements.csv as columns, OOM rows get NaN latency/memory/energy and oom=True
    """
    with open(RESULTS / "measurements.csv") as f:
        rows = list(csv.DictReader(f))
    oom = np.array([r["memory"] == "OOM" for r in rows])

    def col(name: str) -> np.ndarray:
        """
        Parse numeric column, empty and OOM cells become NaN

        Args:
            name: column name
        """
        return np.array([np.nan if r[name] in ("", "OOM") else float(r[name]) for r in rows])

    return {
        "S": np.array([int(r["S"]) for r in rows]),
        "B": np.array([int(r["B"]) for r in rows]),
        "latency": col("latency"),
        "memory": col("memory"),
        "energy": col("energy"),
        "is_validation": np.array([r["is_validation"] == "True" for r in rows]),
        "oom": oom,
    }
