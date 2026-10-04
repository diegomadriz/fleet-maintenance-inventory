# Data contract

The inputs are masked replica tables. Names, dates, identifiers and values use
replica vocabulary. The model command reads the shipped inputs and writes only
outputs. The optional generator writes a separate set of CSVs.

| Input | Grain and purpose |
|---|---|
| `demand_history.csv` | Part × calendar month, including zeros; demand quantities |
| `engine_hours.csv` | Truck × recorded day; daily use and cumulative engine-hour clock |
| `fleet.csv` | Truck; relabelled brand, model year and utilization factor |
| `part_taxonomy.csv` | Part; name, family and corrective scope |
| `replacement_events.csv` | Replacement event; truck, part, units, clock position, age and estimated-hours flag |
| `generator_config.json` | Coarse distribution parameters and replica design settings |

Demand-class labels in input files are generation targets; evaluated classes
are recomputed over the zero-filled calendar panel. Named-part parameters
represent grouped categories. Fleet scope is described as a fleet of 100+
mixer trucks; exact fleet-size metrics are omitted from summary outputs.

## Outputs

| Artifact | Meaning |
|---|---|
| `model_parameters.csv` | Full-calendar Croston parameters, Weibull fits, per-part carrying counts and selected rates; corrective parts with positive in-calendar demand |
| `croston_sba_evaluation.csv` | Fixed training-prefix forecast, holdout errors and evaluated demand class across the whole catalogue, including zero-demand parts |
| `inventory_policy.csv` | Active corrective part × assumed lead time; expected demand, safety stock, minimum and maximum |
| `aft_parameters.csv` | AFT results, original plain-Weibull fallback or no fit; brand coefficients stored as JSON |
| `maintenance_intervals.csv` | Brand × model year × eligible corrective part interval lookup |
| `maintenance_schedule.csv` | Historical corrective truck × part; elapsed hours, confidence and status |
| `summary.json` | Part/policy/pair counts, model and class distributions, and evaluation-window lengths |
| `replica_results.png` | Model-selected reorder-point curves and schedule-status counts |

Blank fit values mean unavailable estimates. Confidence tiers describe interval
sample depth and fallback type. UNKNOWN schedule pairs have no confidence tier.
Zero-demand parts have no policy rows; they can retain maintenance history.
