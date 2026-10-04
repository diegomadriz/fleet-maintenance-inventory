"""Compare a fresh reproduction with saved public tables and summary."""

import json
import sys
from pathlib import Path

import pandas as pd

saved = Path("results")
actual = Path(sys.argv[1])
count = 0
for path in sorted(saved.glob("*.csv")):
    expected = pd.read_csv(path)
    observed = pd.read_csv(actual / path.name)
    pd.testing.assert_frame_equal(expected, observed, check_exact=False, rtol=1e-5, atol=1e-5)
    count += 1
expected = json.loads((saved / "summary.json").read_text())
observed = json.loads((actual / "summary.json").read_text())
assert expected == observed, "Reproduced summary differs from saved replica results"
assert (actual / "replica_results.png").is_file()
print(f"Reproduction comparison: {count} CSVs, summary and figure present; PASS")
