"""Numerical edges, selection gates and model fallback behavior."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fleet_inventory import models
from fleet_inventory.demand import sb_class
from fleet_inventory.pipeline import (
    eligible_intervals,
    forecast_tables,
    monthly_panel,
    policy_table,
    usage_summary,
)


def test_croston_zero_and_state_updates():
    z, p, rate, count = models.croston_sba(pd.Series([0, 0, 0]))
    assert np.isnan(z) and np.isnan(p) and rate == 0 and count == 0
    z, p, rate, count = models.croston_sba(pd.Series([0, 4, 0, 0, 8]), 0.15)
    assert z == pytest.approx(4.6)
    assert p == pytest.approx(1.4275)
    assert rate == pytest.approx(0.925 * 4.6 / 1.4275)
    assert count == 2


@pytest.mark.parametrize("rate", [0, -1, np.nan, np.inf])
def test_invalid_rate_yields_no_stock(rate):
    p = models.compute_minmax(rate, 0.5, 10, 30, stats.norm.ppf(0.95))
    assert p["min"] == p["max"] == p["order_qty"] == 0


def test_stock_compound_variance_and_rounding():
    p = models.compute_minmax(4, 0.5, 15, 30, stats.norm.ppf(0.95))
    assert p["E_demand_lead"] == 2
    assert p["safety_stock"] == pytest.approx(2.601)
    assert (p["min"], p["max"], p["order_qty"]) == (5, 9, 4)
    assert models.compute_minmax(4, np.nan, 15, 30, 1.645) == models.compute_minmax(
        4, 1, 15, 30, 1.645
    )


def test_weibull_six_interval_gate_and_fixed_location(monkeypatch):
    assert np.isnan(models.fit_weibull(np.array([0, -2, np.nan, 2, 3, 4, 5, 6]))[0])
    seen = {}

    def fit(x, floc):
        seen["loc"] = floc
        return 1.2, 0, 800

    monkeypatch.setattr(models.stats.weibull_min, "fit", fit)
    monkeypatch.setattr(models.stats, "kstest", lambda *a, **k: (0.1, 0.08))
    assert models.fit_weibull(np.array([1, 2, 3, 4, 5, 6])) == (1.2, 800, 0.08, 6)
    assert seen == {"loc": 0}


@pytest.mark.parametrize(
    "values,expected",
    [
        ([10] * 24, "smooth"),
        ([1] * 23 + [100], "erratic"),
        ([0, 2] * 12, "lumpy"),
        ([0] * 23 + [1], None),
    ],
)
def test_zero_filled_classification(values, expected):
    assert sb_class(np.array(values, dtype=float)) == expected


@pytest.mark.parametrize(
    "model_type,n,expected",
    [
        (None, 15, "ESTIMATED"),
        ("Weibull_fallback", 30, "LOW"),
        ("AFT", 2, "LOW"),
        ("AFT", 3, "MEDIUM"),
        ("AFT", 9, "MEDIUM"),
        ("AFT", 10, "HIGH"),
    ],
)
def test_confidence_boundaries(model_type, n, expected):
    assert models.confidence_tier({"model_type": model_type}, "Brand X", n) == expected


@pytest.mark.parametrize(
    "pct,expected", [(np.nan, "UNKNOWN"), (0.849, "OK"), (0.85, "DUE SOON"), (1, "OVERDUE")]
)
def test_status_boundaries(pct, expected):
    assert models.status_flag(pct) == expected


def test_expected_eta_has_no_interval_clipping():
    result = {
        "model_type": "AFT",
        "eta_base": 50_000,
        "age_coef": 0.02,
        "brands_in_data": ["Brand X", "Brand Y"],
        "brand_coefs": {"Brand Y": 0.1},
    }
    assert models.expected_eta(result, "Brand Y", 12) == pytest.approx(50_000 * np.exp(0.34))
    assert np.isnan(models.expected_eta(result, "Brand Z", 12))


def test_aft_unpenalized_age_clip_and_original_fallback(monkeypatch):
    calls = {}

    class FailedAFT:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

        def fit(self, frame, **kwargs):
            calls["age"] = frame.Age.tolist()
            raise ValueError("forced fit failure")

    class PlainWeibull:
        rho_ = 2.0
        lambda_ = 400.0

        def fit(self, durations):
            calls["n"] = len(durations)

    monkeypatch.setattr(models, "WeibullAFTFitter", FailedAFT)
    monkeypatch.setattr(models, "WeibullFitter", PlainWeibull)
    frame = pd.DataFrame(
        {"hours_between_failures": [20] * 6, "Age": [-1, 40, 2, 3, 4, 5], "Brand": ["X"] * 6}
    )
    result = models.fit_aft(frame)
    assert calls == {"kwargs": {}, "age": [0, 30, 2, 3, 4, 5], "n": 6}
    assert result["model_type"] == "Weibull_fallback"
    assert result["eta_base"] == pytest.approx(400 ** (-1 / 2))
    # The original fallback leaves age_coef as NaN; downstream Croston fallback is preserved.
    assert np.isnan(models.expected_eta(result, "X", 5))


def test_complete_panel_includes_globally_inactive_month():
    demand = pd.DataFrame(
        {"part_id": ["a", "a"], "period": ["2031-01", "2031-03"], "quantity": [1, 2]}
    )
    config = {"replica_design": {"calendar_start": "2031-01-01", "calendar_end": "2031-03-31"}}
    panel = monthly_panel(demand, pd.DataFrame({"part_id": ["a", "b"]}), config)
    assert panel.loc["a"].tolist() == [1, 0, 2]
    assert panel.loc["b"].tolist() == [0, 0, 0]


def test_zero_demand_parts_are_evaluated_but_excluded_from_policy():
    parts = pd.DataFrame(
        {
            "part_id": ["active", "inactive", "consumable"],
            "part_name": ["Active", "Inactive", "Consumable"],
            "corrective": [True, True, False],
        }
    )
    panel = pd.DataFrame([[1, 0, 2], [0, 0, 0], [1, 2, 3]], index=parts.part_id)
    # History alone cannot put an inactive part into inventory policy, even with a fit.
    intervals = pd.DataFrame(
        {"part_id": ["inactive"] * 6, "hours_between_replacements": [10, 20, 30, 40, 50, 60]}
    )
    events = pd.DataFrame({"part_id": ["inactive"], "truck_id": ["history-only"]})
    settings = {
        "forecast_train_months": 2,
        "croston_alpha": 0.15,
        "lead_times_days": [5, 10, 15, 30],
        "review_days": 30,
        "service_level": 0.95,
    }
    evaluation, parameters = forecast_tables(panel, parts, settings, events, intervals, 100, 1)
    assert set(evaluation.part_id) == set(parts.part_id)
    assert evaluation.set_index("part_id").loc["inactive", "forecast_units_per_month"] == 0
    assert parameters.part_id.tolist() == ["active"]
    policy = policy_table(parameters, settings)
    assert set(policy.part_id) == {"active"}
    assert len(policy) == 4


def test_stability_and_estimated_interval_filter():
    telemetry = pd.DataFrame(
        {
            "truck_id": ["stable"] * 6 + ["short"] * 5,
            "replica_date": list(pd.date_range("2031-01-01", periods=6, freq="MS"))
            + list(pd.date_range("2031-01-01", periods=5, freq="MS")),
            "engine_hours_used": [100] * 11,
        }
    )
    stable, _, _, _ = usage_summary(telemetry, [])
    assert stable == {"stable"}
    events = pd.DataFrame(
        {
            "truck_id": ["stable", "short", "short", "stable", "stable"],
            "hours_estimated": [True, True, False, False, False],
            "hours_between_replacements": [10, 10, 10, 50_000, -1],
        }
    )
    assert eligible_intervals(events, stable).index.tolist() == [0, 2]
