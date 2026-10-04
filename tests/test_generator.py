"""The included generator has no dependency on confidential inputs."""

from pathlib import Path

import pytest

from fleet_inventory.generator import generate

ROOT = Path(__file__).resolve().parents[1]


def test_generator_rejects_existing_directory(tmp_path):
    with pytest.raises(ValueError, match="new directory"):
        generate(
            ROOT / "data/replica/generator_config.json",
            tmp_path,
            ROOT / "config/generator_runtime.json",
        )


def test_generator_recreates_csvs_using_public_config_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "dataset"
    generate(
        ROOT / "data/replica/generator_config.json", output, ROOT / "config/generator_runtime.json"
    )
    for path in output.glob("*.csv"):
        assert path.read_bytes() == (ROOT / "data/replica" / path.name).read_bytes()
    assert len(list(output.glob("*.csv"))) == 5
