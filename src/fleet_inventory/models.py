"""Original modeling functions, isolated from operational loaders and plotting."""

import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter, WeibullFitter
from scipy import stats

MIN_IFI_FOR_WEIBULL = 6
MIN_IFI_AFT = 6
DUE_SOON_PCT = 0.85


def croston_sba(series: np.ndarray, alpha: float = 0.15):
    """
    Syntetos-Boylan Approximation of Croston's method.
    Returns: (z, p, forecast_per_period, n_demand_periods)
      z    = smoothed demand size
      p    = smoothed inter-demand interval (periods)
      rate = SBA forecast = (1 - alpha/2) * z/p
    """
    nonzero = series[series > 0]
    if len(nonzero) == 0:
        return np.nan, np.nan, 0.0, 0

    z = float(nonzero.iloc[0])
    p = 1.0
    q = 1  # periods since last demand

    for x in series:
        if x > 0:
            z = alpha * x + (1 - alpha) * z
            p = alpha * q + (1 - alpha) * p
            q = 1
        else:
            q += 1

    rate = (1 - alpha / 2) * (z / p)
    return z, p, rate, len(nonzero)


def fit_weibull(intervals: np.ndarray):
    """
    MLE 2-parameter Weibull fit (loc fixed at 0).
    Returns (beta, eta, ks_pvalue) or (nan, nan, nan) on failure.
    scipy parameterisation: weibull_min(c=beta, scale=eta, loc=0)
    """
    intervals = intervals[(intervals > 0) & np.isfinite(intervals)]
    if len(intervals) < MIN_IFI_FOR_WEIBULL:
        return np.nan, np.nan, np.nan, len(intervals)
    try:
        beta, loc, eta = stats.weibull_min.fit(intervals, floc=0)
        ks_stat, ks_p = stats.kstest(intervals, "weibull_min", args=(beta, 0, eta))
        return beta, eta, ks_p, len(intervals)
    except Exception:
        return np.nan, np.nan, np.nan, len(intervals)


def compute_minmax(
    rate_per_month: float, cv: float, lead_days: float, review_days: float, z: float
) -> dict:
    """
    Poisson-based min-max with normal safety stock approximation.
      rate_per_month : expected fleet demand per month
      cv             : coefficient of variation of demand size (nan → assume 1)
      lead_days      : supplier lead time in days
      review_days    : replenishment review period in days
    """
    if rate_per_month <= 0 or not np.isfinite(rate_per_month):
        return {"lead_days": lead_days, "min": 0, "max": 0, "safety_stock": 0, "order_qty": 0}

    lead_months = lead_days / 30
    review_months = review_days / 30
    cv_safe = cv if np.isfinite(cv) else 1.0  # default cv=1 if unknown

    E_lead = rate_per_month * lead_months
    E_review = rate_per_month * review_months

    # Variance during lead time: Var[NQ] = E[N]*Var[Q] + Var[N]*E[Q]²
    # Simplified: assume demand qty variance = (cv*mean_size)²
    # Using compound Poisson: std ≈ sqrt(E_lead) * sqrt(1 + cv²)
    std_lead = np.sqrt(E_lead * (1 + cv_safe**2))

    safety = z * std_lead
    min_lvl = np.ceil(E_lead + safety)
    max_lvl = np.ceil(min_lvl + E_review)

    return {
        "lead_days": lead_days,
        "E_demand_lead": round(E_lead, 3),
        "safety_stock": round(safety, 3),
        "min": int(min_lvl),
        "max": int(max_lvl),
        "order_qty": int(max_lvl - min_lvl),
    }


def fit_aft(df: pd.DataFrame):
    """
    Fit WeibullAFT with Brand (dummies) + Age as covariates.
    Falls back to fleet-level Weibull if AFT fails or too few observations.
    Returns dict of fit results.
    """
    df = df.copy()
    df = df[df["hours_between_failures"] > 0].dropna(
        subset=["hours_between_failures", "Brand", "Age"]
    )
    df["Age"] = df["Age"].clip(0, 30)

    result = {
        "n_obs": len(df),
        "brands_in_data": sorted(df["Brand"].unique().tolist()),
        "model_type": None,
        "fitter": None,
        "beta": np.nan,
        "eta_base": np.nan,
        "age_coef": np.nan,
        "age_pval": np.nan,
        "brand_coefs": {},
    }

    if len(df) < MIN_IFI_AFT:
        return result

    # Try AFT with brand dummies + age
    try:
        brands = df["Brand"].unique()
        if len(brands) > 1:
            brand_dummies = pd.get_dummies(df["Brand"], drop_first=True, prefix="brand")
        else:
            brand_dummies = pd.DataFrame(index=df.index)

        fit_df = pd.concat(
            [
                df[["hours_between_failures", "Age"]].reset_index(drop=True),
                brand_dummies.reset_index(drop=True),
            ],
            axis=1,
        )

        aft = WeibullAFTFitter()
        aft.fit(fit_df, duration_col="hours_between_failures")

        # Extract coefficients from lambda_ (scale) model
        coefs = aft.params_["lambda_"]
        result["model_type"] = "AFT"
        result["fitter"] = aft
        result["beta"] = float(np.exp(aft.params_["rho_"]["Intercept"]))
        result["eta_base"] = float(np.exp(coefs.get("Intercept", 0)))
        result["age_coef"] = float(coefs.get("Age", 0))

        # Age p-value
        if hasattr(aft, "summary"):
            try:
                summary = aft.summary
                age_row = summary[summary.index.get_level_values(1) == "Age"]
                result["age_pval"] = float(age_row["p"].iloc[0]) if len(age_row) else np.nan
            except Exception:
                result["age_pval"] = np.nan

        # Brand coefficients (relative to reference brand)
        for col in coefs.index:
            if col.startswith("brand_"):
                brand_name = col.replace("brand_", "")
                result["brand_coefs"][brand_name] = float(coefs[col])

    except Exception:
        # Fallback: plain Weibull (fleet-level for this material)
        try:
            wf = WeibullFitter()
            wf.fit(df["hours_between_failures"])
            result["model_type"] = "Weibull_fallback"
            result["fitter"] = wf
            result["beta"] = float(wf.rho_)
            result["eta_base"] = float(wf.lambda_ ** (-1 / wf.rho_)) if wf.rho_ > 0 else np.nan
        except Exception:
            pass

    return result


def expected_eta(result: dict, brand: str, age: float) -> float:
    """
    Compute expected replacement interval (η) for a given brand and age.
    AFT formula: η = η_base × exp(brand_coef) × exp(age_coef × age)
    """
    if result["model_type"] is None:
        return np.nan

    eta = result["eta_base"]
    if np.isnan(eta):
        return np.nan

    # Apply age effect
    age_coef = result.get("age_coef", 0) or 0
    eta = eta * np.exp(age_coef * float(age))

    # Apply brand effect (reference brand has coef=0)
    brand_coefs = result.get("brand_coefs", {})
    # Find the matching brand coef (drop_first means one brand is reference)
    matched = brand_coefs.get(brand, None)
    if matched is None:
        # Check if it's the reference brand (no coef = 0 adjustment)
        brands_in_data = result.get("brands_in_data", [])
        if brand not in brands_in_data:
            return np.nan
        # It's the reference brand
    else:
        eta = eta * np.exp(matched)

    return eta


def confidence_tier(result: dict, brand: str, brand_n: int) -> str:
    if result["model_type"] is None:
        return "ESTIMATED"
    if result["model_type"] == "Weibull_fallback":
        return "LOW"
    if brand_n >= 10:
        return "HIGH"
    if brand_n >= 3:
        return "MEDIUM"
    return "LOW"


def status_flag(pct):
    if pd.isna(pct):
        return "UNKNOWN"
    if pct >= 1.0:
        return "OVERDUE"
    if pct >= DUE_SOON_PCT:
        return "DUE SOON"
    return "OK"
