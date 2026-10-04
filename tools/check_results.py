"""Compare a fresh reproduction with saved public tables and summary."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RTOL = 1e-5
ATOL = 1e-5

saved = Path("results")
actual = Path(sys.argv[1])
problems = []


def close(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)
    return a == b


def compare_json(a, b, path="summary"):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            compare_json(a.get(key), b.get(key), f"{path}.{key}")
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            compare_json(x, y, f"{path}[{i}]")
    elif not close(a, b):
        problems.append(f"{path}: saved {a!r}, reproduced {b!r}")


count = 0
for csv in sorted(saved.glob("*.csv")):
    expected = pd.read_csv(csv)
    observed = pd.read_csv(actual / csv.name)
    if list(expected.columns) != list(observed.columns) or len(expected) != len(observed):
        problems.append(f"{csv.name}: shape or columns differ")
        continue
    for column in expected.columns:
        e, o = expected[column], observed[column]
        if pd.api.types.is_numeric_dtype(e) and pd.api.types.is_numeric_dtype(o):
            bad = ~np.isclose(e, o, rtol=RTOL, atol=ATOL, equal_nan=True)
            if bad.any():
                worst = float(
                    np.nanmax(np.abs(e[bad] - o[bad]) / np.maximum(np.abs(e[bad]), 1e-12))
                )
                problems.append(
                    f"{csv.name}:{column}: {int(bad.sum())} rows differ, max relative {worst:.2e}"
                )
        elif not e.fillna("").astype(str).equals(o.fillna("").astype(str)):
            problems.append(f"{csv.name}:{column}: text values differ")
    count += 1

compare_json(
    json.loads((saved / "summary.json").read_text()),
    json.loads((actual / "summary.json").read_text()),
)
if not (actual / "replica_results.png").is_file():
    problems.append("replica_results.png missing")

if problems:
    for problem in problems[:40]:
        # GitHub Actions annotation, readable without the raw log.
        print(f"::error::{problem}")
    sys.exit(1)
print(f"Reproduction comparison: {count} CSVs, summary and figure present; PASS")
