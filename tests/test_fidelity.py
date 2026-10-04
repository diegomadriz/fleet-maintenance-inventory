"""Frozen source arithmetic plus independent checks of saved replica outputs."""

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import reference_models
from scipy import stats

from fleet_inventory import models
from fleet_inventory.pipeline import usage_summary

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name",
    [
        "croston_sba",
        "fit_weibull",
        "compute_minmax",
        "fit_aft",
        "expected_eta",
        "confidence_tier",
        "status_flag",
    ],
)
def test_original_function_ast(name):
    actual = ast.parse(inspect.getsource(getattr(models, name)))
    original = ast.parse(inspect.getsource(getattr(reference_models, name)))
    assert ast.dump(actual, include_attributes=False) == ast.dump(
        original, include_attributes=False
    )


def test_croston_matches_frozen_source_on_all_panels():
    demand = pd.read_csv(ROOT / "data/replica/demand_history.csv")
    for _, group in demand.groupby("part_id"):
        series = group.sort_values("period").quantity
        for n in [18, 24]:
            np.testing.assert_allclose(
                models.croston_sba(series.iloc[:n]),
                reference_models.croston_sba(series.iloc[:n]),
                equal_nan=True,
            )


def test_saved_policy_recomputes_from_unrounded_parameters():
    params = pd.read_csv(ROOT / "results/model_parameters.csv").set_index("part_id")
    policy = pd.read_csv(ROOT / "results/inventory_policy.csv")
    for row in policy.itertuples():
        p = params.loc[row.part_id]
        mu = p.policy_rate * row.lead_days / 30
        cv = p.cv_demand_size if np.isfinite(p.cv_demand_size) else 1.0
        safety = stats.norm.ppf(0.95) * np.sqrt(mu * (1 + cv**2))
        minimum = int(np.ceil(mu + safety))
        assert row.min == minimum
        assert row.max == int(np.ceil(minimum + p.policy_rate))
    assert len(policy) == len(params) * 4
    assert set(policy.part_id) == set(params.index)


def test_saved_evaluation_and_summary():
    demand = pd.read_csv(ROOT / "data/replica/demand_history.csv")
    evaluation = pd.read_csv(ROOT / "results/croston_sba_evaluation.csv")
    summary = json.loads((ROOT / "results/summary.json").read_text())
    for row in evaluation.itertuples():
        values = demand[demand.part_id == row.part_id].sort_values("period").quantity
        rate = reference_models.croston_sba(values.iloc[:18])[2]
        errors = values.iloc[18:].to_numpy() - rate
        assert row.forecast_units_per_month == pytest.approx(rate)
        assert row.test_mae == pytest.approx(np.abs(errors).mean())
        assert row.test_rmse == pytest.approx(np.sqrt((errors**2).mean()))
    status = pd.read_csv(ROOT / "results/maintenance_schedule.csv")
    assert summary["parts_evaluated"] == len(evaluation)
    assert summary["maintenance_pairs"] == len(status)
    assert summary["schedule_status_counts"] == status.status.value_counts().to_dict()
    assert summary["confidence_counts"] == status.confidence.value_counts().to_dict()


def test_history_only_and_corrective_scope():
    events = pd.read_csv(ROOT / "data/replica/replacement_events.csv")
    parts = pd.read_csv(ROOT / "data/replica/part_taxonomy.csv")
    status = pd.read_csv(ROOT / "results/maintenance_schedule.csv")
    history = set(zip(events.truck_id, events.part_id))
    assert set(zip(status.truck_id, status.part_id)) <= history
    assert set(status.part_id) <= set(parts.loc[parts.corrective, "part_id"])
    assert not status.duplicated(["truck_id", "part_id"]).any()


def test_policy_scope_excludes_parts_without_calendar_demand():
    demand = pd.read_csv(ROOT / "data/replica/demand_history.csv")
    parts = pd.read_csv(ROOT / "data/replica/part_taxonomy.csv")
    params = pd.read_csv(ROOT / "results/model_parameters.csv")
    policy = pd.read_csv(ROOT / "results/inventory_policy.csv")
    active = set(demand.loc[demand.quantity > 0, "part_id"])
    corrective = set(parts.loc[parts.corrective, "part_id"])
    assert set(params.part_id) == active & corrective
    assert set(policy.part_id) == active & corrective


def test_saved_weibull_selection_and_rate():
    params = pd.read_csv(ROOT / "results/model_parameters.csv")
    telemetry = pd.read_csv(ROOT / "data/replica/engine_hours.csv")
    settings = json.loads((ROOT / "config/model_config.json").read_text())
    _, _, fleet_hours, truck_count = usage_summary(telemetry, settings["excluded_months"])
    for row in params.itertuples():
        preferred = (
            "Weibull"
            if np.isfinite(row.weibull_beta) and row.weibull_ks_pval > 0.05
            else "Croston-SBA"
        )
        assert row.preferred_model == preferred
        if preferred == "Weibull":
            assert row.n_ifi >= 6
            rate = (
                row.n_trucks_part
                * (fleet_hours / truck_count / row.weibull_eta) ** row.weibull_beta
            )
            assert row.policy_rate == pytest.approx(rate)
        else:
            assert row.policy_rate == pytest.approx(row.croston_rate)


def test_saved_schedule_matches_original_coefficients_and_tiers():
    from fleet_inventory.pipeline import eligible_intervals, usage_summary

    events = pd.read_csv(ROOT / "data/replica/replacement_events.csv")
    telemetry = pd.read_csv(ROOT / "data/replica/engine_hours.csv")
    settings = json.loads((ROOT / "config/model_config.json").read_text())
    stable, _, fleet_hours, _ = usage_summary(telemetry, settings["excluded_months"])
    eligible = eligible_intervals(events, stable)
    counts = eligible.groupby(["part_id", "brand"]).size().to_dict()
    models_saved = pd.read_csv(ROOT / "results/aft_parameters.csv").set_index("part_id")
    rates = pd.read_csv(ROOT / "results/model_parameters.csv").set_index("part_id").croston_rate
    schedule = pd.read_csv(ROOT / "results/maintenance_intervals.csv")
    for row in schedule.itertuples():
        result = models_saved.loc[row.part_id].to_dict()
        result["brands_in_data"] = json.loads(result["brands_in_data"])
        result["brand_coefs"] = json.loads(result["brand_coefs"])
        if pd.isna(result["model_type"]):
            result["model_type"] = None
        eta = reference_models.expected_eta(
            result, row.brand, settings["reference_year"] - row.model_year
        )
        tier = reference_models.confidence_tier(
            result, row.brand, counts.get((row.part_id, row.brand), 0)
        )
        if np.isnan(eta) or tier == "ESTIMATED":
            eta = fleet_hours / rates.loc[row.part_id]
            tier = "ESTIMATED"
        assert row.expected_eta_h == round(eta, 0)
        assert row.confidence == tier
    status = pd.read_csv(ROOT / "results/maintenance_schedule.csv")
    for row in status.itertuples():
        assert row.status == reference_models.status_flag(row.pct_consumed)
