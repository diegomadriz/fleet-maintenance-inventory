"""Standalone replica generator using only the published calibration config."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def class_counts(total: int, proportions: dict) -> dict[str, int]:
    names = ["smooth", "erratic", "intermittent", "lumpy"]
    counts = {name: int(round(total * float(proportions.get(name, 0.0)))) for name in names}
    counts["lumpy"] += total - sum(counts.values())
    return counts


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in name.lower()).replace("--", "-").strip("-")


def build_parts(config: dict, rng: np.random.Generator) -> pd.DataFrame:
    design = config["replica_design"]
    named = config.get("named_parts", [])
    shape_low, shape_high = config["reliability"]["weibull_shape_range"]
    scale_low, scale_high = config["reliability"]["weibull_scale_hours_range"]
    rows = []
    for part in named:
        rows.append(
            {
                "part_id": _slug(part["name"]),
                "part_name": part["name"],
                "part_family": "Named example",
                "demand_class": part["demand_class"],
                "corrective": bool(part.get("corrective", True)),
                "true_weibull_shape": part["weibull_shape"]
                or round((shape_low + shape_high) / 2, 3),
                "true_weibull_scale_hours": part["weibull_scale_hours"]
                or round((scale_low + scale_high) / 2, 1),
            }
        )
    generic_total = design["part_count"] - len(rows)
    tail = config.get("rare_tail")
    corrective_share = config.get("corrective_share_by_class", {})

    def generic_counts(rare_total: int) -> dict:
        # The real class mix holds over the whole catalogue: named and rare parts take their classes first,
        # the generic parts make up the rest of each class.
        counts = class_counts(
            design["part_count"] + rare_total, config["demand"]["class_proportions"]
        )
        for klass in [p["demand_class"] for p in named] + ["lumpy"] * rare_total:
            counts[klass] = counts.get(klass, 0) - 1
        counts = {k: max(0, v) for k, v in counts.items()}
        counts["lumpy"] += generic_total - sum(counts.values())
        return counts

    # The rare tail is a share of the scheduled (corrective) parts, so size it against them.
    rare_total, counts = 0, generic_counts(0)
    if tail:
        share = float(tail["share_of_scheduled_parts"])
        for _ in range(5):
            scheduled = sum(bool(p.get("corrective", True)) for p in named) + sum(
                round(n * float(corrective_share.get(k, 1.0))) for k, n in counts.items()
            )
            rare_total = int(round(scheduled * share / (1 - share)))
            counts = generic_counts(rare_total)
    classes = [
        name for name in ("smooth", "erratic", "intermittent", "lumpy") for _ in range(counts[name])
    ]
    rng.shuffle(classes)
    seen: dict[str, int] = {}
    for idx, demand_class in enumerate(classes):
        position = idx / max(1, len(classes) - 1)
        seen[demand_class] = seen.get(demand_class, 0) + 1
        # The first share of each class's generic parts are mechanical corrective; the rest are consumables.
        corrective = seen[demand_class] <= round(
            counts[demand_class] * float(corrective_share.get(demand_class, 1.0))
        )
        rows.append(
            {
                "part_id": f"part-{idx + 1:02d}",
                "part_name": f"Other part {idx + 1:02d}",
                "part_family": "Other",
                "demand_class": demand_class,
                "corrective": bool(corrective),
                "true_weibull_shape": round(shape_low + position * (shape_high - shape_low), 3),
                "true_weibull_scale_hours": round(
                    scale_low
                    + ((idx * 7) % len(classes))
                    / max(1, len(classes) - 1)
                    * (scale_high - scale_low),
                    1,
                ),
            }
        )
    # Original catalogue shape: a tail of rarely replaced parts, too sparse for a reliability fit.
    if tail:
        for idx in range(rare_total):
            rows.append(
                {
                    "part_id": f"rare-{idx + 1:02d}",
                    "part_name": f"Rare part {idx + 1:02d}",
                    "part_family": "Rare",
                    "demand_class": "lumpy",
                    "corrective": True,
                    "true_weibull_shape": round(
                        shape_low + rng.random() * (shape_high - shape_low), 3
                    ),
                    "true_weibull_scale_hours": round(scale_high, 1),
                }
            )
    return pd.DataFrame(rows)


def build_fleet(config: dict, rng: np.random.Generator) -> pd.DataFrame:
    design = config["replica_design"]
    brand_probs = np.array(design["brand_shares"], dtype=float)
    brand_probs = brand_probs / brand_probs.sum()
    brands = rng.choice(design["brands"], design["fleet_size"], p=brand_probs)
    model_year = rng.integers(2018, 2031, design["fleet_size"])
    utilization_factor = rng.lognormal(
        mean=-0.5 * config["telemetry"]["truck_heterogeneity_cv"] ** 2,
        sigma=config["telemetry"]["truck_heterogeneity_cv"],
        size=design["fleet_size"],
    )
    return pd.DataFrame(
        {
            "truck_id": [f"Truck {i:03d}" for i in range(1, design["fleet_size"] + 1)],
            "brand": brands,
            "model_year": model_year,
            "utilization_factor": utilization_factor.round(4),
            "initial_engine_hours": rng.integers(1_000, 16_001, design["fleet_size"]),
        }
    )


def generate_telemetry(
    config: dict, fleet: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range(
        config["replica_design"]["calendar_start"],
        config["replica_design"]["calendar_end"],
        freq="D",
    )
    center = config["telemetry"]["positive_daily_hours_center"]
    spread = config["telemetry"]["daily_hours_spread"]
    shape = max((center / max(spread, 0.1)) ** 2, 0.5)
    scale = max(spread**2 / max(center, 0.1), 0.1)
    records = []
    full_records = []
    for truck in fleet.itertuples(index=False):
        raw = rng.gamma(shape, scale, len(dates)) * truck.utilization_factor
        weekday = np.where(
            dates.dayofweek < 5,
            config["telemetry"]["weekday_multiplier"],
            config["telemetry"]["weekend_multiplier"],
        )
        seasonal = 1.0 + 0.08 * np.sin(2 * np.pi * (dates.dayofyear.to_numpy() / 365.25))
        idle = rng.random(len(dates)) < 0.10
        # Month-to-month irregularity like the real fleet (idle spells, heavy months); a calibrated share of
        # trucks stays steady (monthly CV below 0.5), which is what the original's stability rule selects.
        target_cv = config["telemetry"].get("monthly_cv_median", 0.0)
        month_index = (dates.year - dates.year.min()) * 12 + dates.month - 1
        if target_cv > 0:
            steady = rng.random() < config["telemetry"].get("stable_share", 0.5)
            cv_t = rng.uniform(0.15, 0.4) if steady else target_cv * rng.lognormal(0.15, 0.2)
            sigma = math.sqrt(math.log(1 + cv_t**2))
            monthly = rng.lognormal(-(sigma**2) / 2, sigma, month_index.max() + 1)
            raw = raw * monthly[month_index]
        hours = np.where(idle, 0.0, raw * weekday * seasonal).clip(0, 18)
        cumulative = truck.initial_engine_hours + np.cumsum(hours)
        missing = rng.random(len(dates)) < config["telemetry"]["missing_rate"]
        for date, used, total, is_missing in zip(dates, hours, cumulative, missing):
            row = {
                "truck_id": truck.truck_id,
                "replica_date": date.strftime("%Y-%m-%d"),
                "engine_hours_used": round(float(used), 2),
                "cumulative_engine_hours": round(float(total), 2),
                "telemetry_observed": not bool(is_missing),
            }
            full_records.append(row)
            if not is_missing:
                records.append(row)
    return pd.DataFrame(records), pd.DataFrame(full_records)


def generate_renewals(
    config: dict,
    parts: pd.DataFrame,
    fleet: pd.DataFrame,
    full_telemetry: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replacements as a renewal process on each truck's engine-hour clock.

    As in the original data, a replacement consumes units (the demand line) and the interval
    between two replacements of a part on the same truck is the difference of that truck's
    cumulative engine hours (cumulative minus previous cumulative). Monthly demand is the sum of
    units replaced, so demand, intervals and the maintenance schedule all describe the same events.
    """
    reliability = config["reliability"]
    history_days = config["replica_runtime"]["history_days"]
    base = config["installed_base"]
    named = {_slug(item["name"]): item for item in config.get("named_parts", [])}
    dates = pd.to_datetime(full_telemetry["replica_date"].unique())
    dates = np.array(sorted(dates))
    clock = {
        t: g.sort_values("replica_date")["cumulative_engine_hours"].to_numpy(float)
        for t, g in full_telemetry.groupby("truck_id")
    }
    months_total = len(pd.period_range(dates[0], dates[-1], freq="M"))
    fleet_lookup = fleet.set_index("truck_id")
    usage_center = fleet["utilization_factor"].mean()
    truck_ids = fleet["truck_id"].to_numpy()
    end_year = pd.Timestamp(config["replica_design"]["calendar_end"]).year
    rows, event_number = [], 1
    for part in parts.itertuples(index=False):
        params = dict(
            named.get(part.part_id) or config["demand"]["class_parameters"][part.demand_class]
        )
        klass = part.demand_class
        rare = part.part_family == "Rare"
        units = float(params.get("units_per_replacement") or base["units_per_replacement"])
        trucks_base = (
            config["rare_tail"]["trucks_with_part"]
            if rare
            else params.get("trucks_with_part") or base["trucks_with_part"]
        )
        n_trucks = int(
            np.clip(
                round(
                    float(trucks_base)
                    * (
                        1.0
                        if not rare and params.get("trucks_with_part")
                        else rng.lognormal(0, 0.4)
                    )
                ),
                2,
                len(truck_ids),
            )
        )
        active_share = 1.0 - float(params["zero_probability"])
        target = max(
            active_share * float(params["active_size_mean"]), 0.2
        )  # units per month for the fleet
        if klass == "smooth":
            target = max(target, 6.0)
        events_per_month = target / units
        chosen = rng.choice(truck_ids, n_trucks, replace=False)
        shape = float(part.true_weibull_shape)
        # One engine-hour life scale per part: a part's life is measured in engine hours, so heavy-use
        # trucks reach it sooner in calendar time rather than having longer lives.
        fleet_per_month = float(
            np.mean([(clock[t][-1] - clock[t][0]) / months_total for t in chosen])
        )
        part_scale = fleet_per_month * n_trucks / events_per_month / math.gamma(1 + 1 / shape)
        if rare:
            # About one replacement per truck across history and calendar: too few intervals to fit.
            part_scale = (
                fleet_per_month
                * (months_total + history_days / 30.4)
                / 1.2
                / math.gamma(1 + 1 / shape)
            )
        for truck_id in chosen:
            hours = clock[truck_id]
            scale = part_scale
            truck = fleet_lookup.loc[truck_id]
            brand_effect = reliability["brand_log_time_effects"][truck["brand"]]
            utilization = float(truck["utilization_factor"])
            age = max(1.0, end_year - float(truck["model_year"]))
            scale *= math.exp(
                reliability["replica_age_log_time_coefficient"] * age
                + reliability["replica_usage_log_time_coefficient"]
                * ((utilization - usage_center) / 0.25)
                + brand_effect
            )
            # As in the original, replacement history predates telemetry: simulate the configured history window earlier
            # on a back-extrapolated engine-hour clock, flagging those events as estimated.
            rate_per_day = (hours[-1] - hours[0]) / max(1, len(hours) - 1)
            start_hours = hours[0] - rate_per_day * history_days
            threshold = start_hours + rng.random() * scale * rng.weibull(shape)
            previous = None
            while True:
                if threshold < hours[0]:
                    back = int(math.ceil((hours[0] - threshold) / max(rate_per_day, 1e-6)))
                    date = pd.Timestamp(dates[0]) - pd.Timedelta(days=back)
                    cumulative, estimated = round(float(hours[0] - back * rate_per_day), 2), True
                    idx = None
                else:
                    idx = int(np.searchsorted(hours, threshold))
                    if idx >= len(hours):
                        break
                    cumulative, estimated = float(hours[idx]), False
                if klass in ("lumpy", "erratic"):
                    bulk = rng.random() < 0.3
                    size = max(
                        1,
                        int(
                            round(
                                rng.gamma(2.0, units / 2.0)
                                * (rng.uniform(2.5, 5.0) if bulk else 1.0)
                            )
                        ),
                    )
                else:
                    size = max(1, int(round(units + rng.normal(0, 0.3))))
                if idx is not None:
                    date = pd.Timestamp(dates[idx])
                rows.append(
                    {
                        "event_id": f"REP-EVT-{event_number:05d}",
                        "truck_id": truck_id,
                        "part_id": part.part_id,
                        "part_family": part.part_family,
                        "brand": truck["brand"],
                        "replica_event_date": date.strftime("%Y-%m-%d"),
                        "truck_age_years": round(
                            max(1.0, date.year - float(truck["model_year"])), 2
                        ),
                        "utilization_index": round(utilization, 4),
                        "hours_between_replacements": round(cumulative - previous, 2)
                        if previous is not None
                        else None,
                        "cumulative_engine_hours": round(cumulative, 2),
                        "units_replaced": size,
                        "hours_estimated": estimated,
                        "event_observed": 1,
                    }
                )
                event_number += 1
                previous = cumulative
                threshold = cumulative + scale * rng.weibull(shape)
    events = pd.DataFrame(rows).sort_values(["replica_event_date", "event_id"])
    periods = pd.period_range(
        pd.Timestamp(dates[0]).to_period("M"), pd.Timestamp(dates[-1]).to_period("M"), freq="M"
    )
    in_calendar = events[~events["hours_estimated"]]
    monthly = (
        in_calendar.assign(
            period=pd.to_datetime(in_calendar["replica_event_date"]).dt.to_period("M")
        )
        .groupby(["part_id", "period"])["units_replaced"]
        .sum()
    )
    demand_rows = []
    for part in parts.itertuples(index=False):
        for period in periods:
            demand_rows.append(
                {
                    "part_id": part.part_id,
                    "part_family": part.part_family,
                    "demand_class": part.demand_class,
                    "period": str(period),
                    "quantity": int(monthly.get((part.part_id, period), 0)),
                }
            )
    return events, pd.DataFrame(demand_rows)


def generate(config_path: Path, output: Path, runtime_path=Path("config/generator_runtime.json")):
    """Generate a separate exploration dataset; never replace the shipped inputs."""
    config = json.loads(config_path.read_text())
    config["replica_runtime"] = json.loads(runtime_path.read_text())
    if output.exists():
        raise ValueError("Generator output must be a new directory")
    rng = np.random.default_rng(config["replica_design"]["random_seed"])
    parts = build_parts(config, rng)
    fleet = build_fleet(config, rng)
    telemetry, full_telemetry = generate_telemetry(config, fleet, rng)
    events, demand = generate_renewals(config, parts, fleet, full_telemetry, rng)
    output.mkdir(parents=True)
    for name, frame in {
        "part_taxonomy": parts.drop(columns=["true_weibull_shape", "true_weibull_scale_hours"]),
        "fleet": fleet.drop(columns=["initial_engine_hours"]),
        "engine_hours": telemetry,
        "replacement_events": events,
        "demand_history": demand,
    }.items():
        frame.to_csv(output / f"{name}.csv", index=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("data/replica/generator_config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runtime-config", type=Path, default=Path("config/generator_runtime.json")
    )
    args = parser.parse_args()
    generate(args.config, args.output, args.runtime_config)
    print("Generated standalone exploration dataset")


if __name__ == "__main__":
    main()
