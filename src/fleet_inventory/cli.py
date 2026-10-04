"""Run the models using only the masked replica shipped in this checkout."""

import argparse
import json
from pathlib import Path

from .pipeline import reproduce


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Public checkout root")
    parser.add_argument("--output", type=Path, help="Result directory; default: ROOT/results")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = (args.output or root / "results").resolve()
    if output == root / "data" or root / "data" in output.parents:
        parser.error("Results must be separate from immutable replica inputs")
    result = reproduce(root, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
