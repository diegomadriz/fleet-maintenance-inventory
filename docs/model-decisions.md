# Modeling decisions

The pipeline connects calendar-month spare-parts demand to replacement intervals on
an engine-hour clock. Inventory and maintenance models cover corrective parts;
a separate fixed-forecast evaluation covers the whole catalogue.

## Demand and inventory

Calendar completion retains zero months. Parts with no positive in-calendar demand
remain in the evaluation but have no model-parameter or inventory-policy rows,
matching the original policy population. Earlier replacement history alone does
not qualify a part for inventory. Full-calendar Croston-SBA estimates feed policy;
the training prefix supplies forecasts held fixed over the holdout.

Croston-SBA uses alpha 0.15. Demand classification uses ADI and CV² over the
zero-filled panel, with cutoffs 1.32 and 0.49 and at least two positive months.
Weibull fitting fixes location at zero, requires six eligible intervals and becomes
preferred only when its KS p-value exceeds 0.05. Eligible intervals are positive,
finite and below 50,000 hours. Estimated intervals enter fitting only for trucks
whose usage CV is below 0.5 over at least six positive monthly observations.

For Weibull-preferred parts, the monthly demand approximation is
`carrying_trucks * (monthly_hours_per_truck / eta) ** beta`. This is the original
approximation, rather than a renewal rate. Min-max policy uses a 95% service target,
assumed lead times of 5, 10, 15 and 30 days, and a 30-day review period. These are
scenario settings, not measured service outcomes. Full precision is retained
until stock rounding.

The masked replica selects Weibull for far more parts than the real project did:
the model-selection share gap is 42 percentage points in a comparison covering
a different population from this pipeline's active corrective-part output. Its
model mix is not representative of the real project. The stock figure uses each part's selected
model rate; the portfolio walkthrough uses Croston-SBA forecasts, so reorder-point
examples can differ. Holdout-error agreement and model-ranking agreement with
real data have not been established.

## AFT and schedule behavior

The original unpenalized `WeibullAFTFitter()` uses age and brand covariates, with
fitting ages clipped to 0–30. Prediction applies
`eta_base * exp(age_coef * age) * exp(brand_coef)` without interval clipping.
An unknown brand triggers a Croston fallback. Confidence tiers describe sample
depth and fallback source, rather than calibrated prediction coverage.

The pipeline keeps the original modeling choices: no AFT penalizer, unbounded
interval predictions, and the original plain-Weibull fallback scale
`lambda_ ** (-1 / rho_)` with an unset age coefficient.

The inverse-scale fallback is preserved despite differing from the usual
lifelines characteristic-life parameterization. Its unset age coefficient can
propagate NaN and cause downstream Croston estimation. A forced-fit-failure test
covers this behavior. Changing it would change model results.

Intervals are rounded to whole hours before schedule status is calculated.
Elapsed hours are floored at zero, consumed ratio is clipped to 0–2, DUE SOON
starts at 0.85 and OVERDUE at 1.0. Missing intervals produce UNKNOWN.
The brand/year interval lookup does not assign parts to trucks: final schedule
rows require a recorded corrective replacement for that truck–part pair.
Parts without eligible intervals do not enter the interval lookup. UNKNOWN rows
have no confidence tier.

## Explicit replica inputs

`config/model_config.json` makes the observation calendar, reference year and
scenario assumptions explicit. The replica uses no excluded months. Positive
monthly telemetry sums determine stability and exposure. Monthly fleet-hour
means and active-truck counts replace private hardcoded operational constants;
per-part carrying counts come from replacement history. The original Croston
interval fallback `(fleet_hours / carrying_trucks) / (rate / carrying_trucks)`
is unchanged.

The prediction year comes from the replica calendar. Event ages are replica
inputs. Current hours use each truck's maximum recorded replica clock instead of
a private lifetime snapshot. Earlier history supplies interval fitting and
last-event clocks; the complete calendar panel supplies forecasts. These inputs
do not establish real-data forecast accuracy or schedule equivalence.

## Generator and numerical environment

The optional generator uses only the shipped coarse configuration and
`config/generator_runtime.json`. Its 1,460-day history window is a four-year
design setting. It writes to a new directory and refuses an existing output
path. It does not run during model reproduction.

```bash
python -m fleet_inventory.generator --output out/exploration
```

The numerical environment and CI use Python 3.12 with pinned NumPy, SciPy,
pandas, lifelines, Matplotlib, autograd and formulaic dependencies. Other Python
versions are outside the declared support range. Optimizer results may differ
on other numerical stacks.
