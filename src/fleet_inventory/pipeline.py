"""Replica-only file pipeline; operational ingestion is intentionally separate."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .demand import sb_class
from .models import (
    compute_minmax,
    confidence_tier,
    croston_sba,
    expected_eta,
    fit_aft,
    fit_weibull,
    status_flag,
)


def monthly_panel(demand, parts, config):
    """Complete part-month grain, preserving inactive months as zero."""
    design = config["replica_design"]
    periods = pd.period_range(design["calendar_start"], design["calendar_end"], freq="M")
    index = pd.MultiIndex.from_product(
        [parts.part_id, periods.astype(str)], names=["part_id", "period"]
    )
    return (
        demand.groupby(["part_id", "period"]).quantity.sum().reindex(index, fill_value=0).unstack()
    )


def usage_summary(telemetry, excluded_months):
    usage = telemetry.assign(
        period=pd.to_datetime(telemetry.replica_date).dt.to_period("M").astype(str)
    )
    usage = usage[~usage.period.isin(excluded_months)]
    monthly = usage.groupby(["truck_id", "period"]).engine_hours_used.sum()
    positive = monthly[monthly > 0]
    cv = (
        positive.groupby(level=0)
        .agg(lambda x: x.std() / x.mean() if len(x) >= 6 else np.nan)
        .dropna()
    )
    stable = set(cv[cv < 0.5].index)
    fleet_hours = float(monthly.groupby(level=1).sum().mean())
    truck_count = int(usage.groupby("period").truck_id.nunique().median())
    return stable, cv, fleet_hours, truck_count


def eligible_intervals(events, stable):
    hours = events.hours_between_replacements
    return events[
        hours.notna()
        & (hours > 0)
        & (hours < 50_000)
        & (~events.hours_estimated | events.truck_id.isin(stable))
    ].copy()


def forecast_tables(panel, parts, settings, events, intervals, fleet_hours, truck_count):
    evaluation, parameters = [], []
    carrying = events.groupby("part_id").truck_id.nunique().to_dict()
    for part in parts.itertuples(index=False):
        series = panel.loc[part.part_id].astype(float)
        n_train = settings["forecast_train_months"]
        if len(series) <= n_train:
            raise ValueError("The replica must supply a nonempty forecast holdout")
        size, elapsed, forecast, _ = croston_sba(series.iloc[:n_train], settings["croston_alpha"])
        actual = series.iloc[n_train:].to_numpy()
        errors = actual - forecast
        evaluation.append(
            {
                "part_id": part.part_id,
                "part_name": part.part_name,
                "corrective": part.corrective,
                "demand_class": sb_class(series.to_numpy()),
                "train_months": n_train,
                "test_months": len(actual),
                "croston_size": size,
                "croston_interval": elapsed,
                "forecast_units_per_month": forecast,
                "test_mae": float(np.abs(errors).mean()),
                "test_rmse": float(np.sqrt((errors**2).mean())),
                "test_bias": float((forecast - actual).mean()),
                "test_total_actual": float(actual.sum()),
            }
        )
        if not part.corrective or not (series > 0).any():
            continue
        size, elapsed, rate, active = croston_sba(series, settings["croston_alpha"])
        positive = series[series > 0]
        cv = float(positive.std() / positive.mean()) if len(positive) > 1 else np.nan
        values = intervals.loc[
            intervals.part_id == part.part_id, "hours_between_replacements"
        ].to_numpy()
        beta, eta, ks_p, n = fit_weibull(values)
        n_trucks = carrying.get(part.part_id, 1)
        weibull_rate = (
            (fleet_hours / truck_count / eta) ** beta * n_trucks if np.isfinite(eta) else np.nan
        )
        preferred = "Weibull" if np.isfinite(beta) and ks_p > 0.05 else "Croston-SBA"
        parameters.append(
            {
                "part_id": part.part_id,
                "part_name": part.part_name,
                "croston_z": size,
                "croston_p": elapsed,
                "croston_rate": rate,
                "n_demand_months": active,
                "total_months": len(series),
                "cv_demand_size": cv,
                "weibull_beta": beta,
                "weibull_eta": eta,
                "weibull_ks_pval": ks_p,
                "n_ifi": n,
                "n_trucks_part": n_trucks,
                "weibull_fleet_rate": weibull_rate,
                "preferred_model": preferred,
                "policy_rate": weibull_rate if preferred == "Weibull" else rate,
            }
        )
    return pd.DataFrame(evaluation), pd.DataFrame(parameters)


def policy_table(parameters, settings):
    rows = []
    for row in parameters.itertuples(index=False):
        for lead in settings["lead_times_days"]:
            policy = compute_minmax(
                row.policy_rate,
                row.cv_demand_size,
                lead,
                settings["review_days"],
                stats.norm.ppf(settings["service_level"]),
            )
            rows.append(
                {
                    "part_id": row.part_id,
                    "preferred_model": row.preferred_model,
                    "policy_rate": row.policy_rate,
                    **policy,
                }
            )
    return pd.DataFrame(rows)


def maintenance_tables(
    events,
    intervals,
    fleet,
    telemetry,
    parameters,
    settings,
    fleet_hours,
    truck_count,
    combo_history=None,
):
    """Original brand/year schedule joined to pairs with a replacement on record."""
    saved, schedule = [], []
    rates = parameters.set_index("part_id").croston_rate.to_dict()
    carrying = events.groupby("part_id").truck_id.nunique().to_dict()
    combo_history = events if combo_history is None else combo_history
    fleet_combos = (
        combo_history.merge(fleet[["truck_id", "model_year"]], on="truck_id")[
            ["brand", "model_year"]
        ]
        .drop_duplicates()
        .sort_values(["brand", "model_year"])
    )
    for part_id, group in intervals.groupby("part_id", sort=True):
        original_fields = group.rename(
            columns={
                "hours_between_replacements": "hours_between_failures",
                "brand": "Brand",
                "truck_age_years": "Age",
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_aft(original_fields)
        saved.append({"part_id": part_id, **{k: v for k, v in result.items() if k != "fitter"}})
        brand_counts = group.groupby("brand").size()
        for combo in fleet_combos.itertuples(index=False):
            eta = expected_eta(result, combo.brand, settings["reference_year"] - combo.model_year)
            brand_n = int(brand_counts.get(combo.brand, 0))
            tier = confidence_tier(result, combo.brand, brand_n)
            if np.isnan(eta) or tier == "ESTIMATED":
                rate = rates.get(part_id, 0)
                if rate > 0:
                    n_trucks = carrying.get(part_id, truck_count)
                    eta = (fleet_hours / n_trucks) / (rate / n_trucks)
                    tier = "ESTIMATED"
            if np.isnan(eta) or eta <= 0:
                continue
            schedule.append(
                {
                    "part_id": part_id,
                    "brand": combo.brand,
                    "model_year": combo.model_year,
                    "expected_eta_h": round(eta, 0),
                    "confidence": tier,
                    "brand_n_ifi": brand_n,
                    "model_type": result["model_type"] or "ESTIMATED",
                }
            )
    intervals_out = pd.DataFrame(
        schedule,
        columns=[
            "part_id",
            "brand",
            "model_year",
            "expected_eta_h",
            "confidence",
            "brand_n_ifi",
            "model_type",
        ],
    )
    last = (
        events[events.cumulative_engine_hours.notna()]
        .groupby(["truck_id", "part_id"])
        .agg(
            last_event_date=("replica_event_date", "max"),
            last_hours=("cumulative_engine_hours", "max"),
            brand=("brand", "first"),
            n_replacements=("units_replaced", "count"),
        )
        .reset_index()
    )
    last = last.merge(fleet[["truck_id", "model_year"]], on="truck_id", how="left")
    current = telemetry.groupby("truck_id").cumulative_engine_hours.max()
    last["current_hours"] = last.truck_id.map(current)
    last["hours_since_last"] = (last.current_hours - last.last_hours).clip(lower=0)
    last = last.merge(
        intervals_out, on=["part_id", "brand", "model_year"], how="left", validate="many_to_one"
    )
    last["pct_consumed"] = (last.hours_since_last / last.expected_eta_h).clip(0, 2)
    last["status"] = last.pct_consumed.apply(status_flag)
    last["hours_remaining"] = (last.expected_eta_h - last.hours_since_last).round(0)
    last = last.sort_values(["status", "pct_consumed"], ascending=[True, False])
    return pd.DataFrame(saved), intervals_out, last


def figure(output, policy, status):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for part in ["air-valve", "battery", "drive-tire"]:
        rows = policy[policy.part_id == part]
        axes[0].plot(rows.lead_days, rows["min"], marker="o", label=part.replace("-", " ").title())
    axes[0].set(
        xlabel="Assumed lead time (days)",
        ylabel="Reorder point (units)",
        title="Replica inventory scenarios",
    )
    axes[0].legend(frameon=False)
    counts = status.status.value_counts().reindex(
        ["OK", "DUE SOON", "OVERDUE", "UNKNOWN"], fill_value=0
    )
    axes[1].bar(counts.index, counts.values, color=["#217a64", "#dba13a", "#b54442", "#89939d"])
    axes[1].set(ylabel="Truck-part pairs", title="Replica maintenance signals")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "replica_results.png", dpi=150, metadata={"Software": "Matplotlib"})
    plt.close(fig)


def reproduce(root: Path, output: Path):
    data = root / "data/replica"
    config = json.loads((data / "generator_config.json").read_text())
    settings = json.loads((root / "config/model_config.json").read_text())
    parts = pd.read_csv(data / "part_taxonomy.csv")
    demand = pd.read_csv(data / "demand_history.csv")
    events = pd.read_csv(data / "replacement_events.csv")
    telemetry = pd.read_csv(data / "engine_hours.csv")
    fleet = pd.read_csv(data / "fleet.csv")
    panel = monthly_panel(demand, parts, config)
    stable, _, fleet_hours, truck_count = usage_summary(telemetry, settings["excluded_months"])
    corrective = set(parts.loc[parts.corrective, "part_id"])
    events_cm = events[events.part_id.isin(corrective)]
    intervals = eligible_intervals(events_cm, stable)
    evaluation, parameters = forecast_tables(
        panel, parts, settings, events_cm, intervals, fleet_hours, truck_count
    )
    policy = policy_table(parameters, settings)
    aft, schedule_intervals, status = maintenance_tables(
        events_cm,
        intervals,
        fleet,
        telemetry,
        parameters,
        settings,
        fleet_hours,
        truck_count,
        combo_history=events,
    )
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "croston_sba_evaluation": evaluation,
        "model_parameters": parameters,
        "inventory_policy": policy,
        "aft_parameters": aft,
        "maintenance_intervals": schedule_intervals,
        "maintenance_schedule": status,
    }
    for name, table in tables.items():
        if name == "aft_parameters":
            table = table.copy()
            for col in ["brands_in_data", "brand_coefs"]:
                table[col] = table[col].map(lambda v: json.dumps(v, sort_keys=True))
        table.to_csv(output / f"{name}.csv", index=False)
    classified = evaluation.demand_class.value_counts().to_dict()
    summary = {
        "label": "replica results",
        "parts_evaluated": len(evaluation),
        "corrective_parts_modeled": len(parameters),
        "inventory_policy_rows": len(policy),
        "maintenance_pairs": len(status),
        "schedule_status_counts": status.status.value_counts().to_dict(),
        "confidence_counts": status.confidence.value_counts().to_dict(),
        "demand_class_counts": classified,
        "preferred_model_counts": parameters.preferred_model.value_counts().to_dict(),
        "aft_model_counts": aft.model_type.fillna("no_fit").value_counts().to_dict(),
        "train_months": settings["forecast_train_months"],
        "holdout_months": int(evaluation.test_months.iloc[0]),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    figure(output, policy, status)
    return summary
