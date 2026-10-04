"""End-to-end CLI coverage with the iterative AFT fitting replaced by a deterministic fake."""

import json
from pathlib import Path

import numpy as np
import pytest

from fleet_inventory import cli, pipeline

ROOT = Path(__file__).resolve().parents[1]


def test_main_command(tmp_path, monkeypatch, capsys):
    def fake_fit(frame):
        return {
            "n_obs": len(frame),
            "brands_in_data": sorted(frame.Brand.unique()),
            "model_type": "AFT" if len(frame) >= 6 else None,
            "fitter": None,
            "beta": 1.2,
            "eta_base": 1_500.0,
            "age_coef": 0.0,
            "age_pval": np.nan,
            "brand_coefs": {},
        }

    monkeypatch.setattr(pipeline, "fit_aft", fake_fit)
    cli.main(["--root", str(ROOT), "--output", str(tmp_path / "results")])
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads((tmp_path / "results/summary.json").read_text())
    assert printed["inventory_policy_rows"] == 156
    assert printed["parts_evaluated"] == 54
    assert (tmp_path / "results/replica_results.png").is_file()
    assert len(list((tmp_path / "results").glob("*.csv"))) == 6


def test_cli_protects_inputs():
    with pytest.raises(SystemExit):
        cli.main(["--root", str(ROOT), "--output", str(ROOT / "data/replica")])
