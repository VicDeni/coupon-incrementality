"""Phase 3c - staggered event study (Callaway & Sant'Anna, 2021).

Households enter the coupon program in different weeks. A standard two-way fixed effects
regression would use already-treated households as controls and can get the sign wrong.
Callaway-Sant'Anna avoids that: each cohort (households whose first campaign starts in
week g) is compared only with households that have *not yet* received any campaign.

For cohort g and week t the building block is a 2x2 difference-in-differences

    ATT(g, t) = E[Y_t - Y_base | cohort g] - E[Y_t - Y_base | not yet treated at t]

with base period = the average of the last weeks before g. With `covariates=True` the
control group's change is first regressed on pre-campaign covariates and predicted for the
cohort (outcome-regression adjustment), so the comparison is between households with the
same baseline spending. ATT(g, t) is then averaged over cohorts by event time e = t - g,
weighting by cohort size. Inference is a Bayesian bootstrap over households.

The estimator is written out in numpy (about 80 lines) so that it can be unit-tested
against simulated data with a known answer; see tests/test_event_study.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .config import FIRST_FULL_WEEK, LAST_WEEK, SEED, save_result
from .db import query

NEVER = 10_000  # cohort code for households that never receive a campaign
BASE_WEEKS = 4  # weeks averaged to form the pre-treatment base


def _wls_predict(x_fit, y_fit, w_fit, x_new):
    """Weighted least squares fit on (x_fit, y_fit), predictions at x_new."""
    root = np.sqrt(w_fit)[:, None]
    beta, *_ = np.linalg.lstsq(x_fit * root, y_fit * root[:, 0], rcond=None)
    return x_new @ beta


def att_by_event_time(
    y: np.ndarray,
    cohort: np.ndarray,
    weeks: np.ndarray,
    event_window: tuple[int, int],
    x: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> pd.Series:
    """Cohort-size-weighted average of ATT(g, t) by event time.

    y       : households x weeks outcome matrix
    cohort  : first treated week per household (NEVER if never treated)
    weeks   : week label of each column of y
    x       : optional households x weeks x k array; x[:, j] holds covariates measured
              before week weeks[j] (used when that week is a cohort's start)
    weights : optional household weights (used by the bootstrap)
    """
    weights = np.ones(len(cohort)) if weights is None else weights
    column = {week: j for j, week in enumerate(weeks)}
    low, high = event_window
    totals: dict[int, float] = {}
    mass: dict[int, float] = {}

    for g in np.unique(cohort[cohort != NEVER]):
        base_weeks = [w for w in range(g - BASE_WEEKS, g) if w in column]
        if not base_weeks:
            continue
        treated = cohort == g
        size = weights[treated].sum()
        base = y[:, [column[w] for w in base_weeks]].mean(axis=1)
        for e in range(low, high + 1):
            t = g + e
            if t not in column or t in base_weeks:
                continue
            controls = cohort > max(t, g)  # includes never-treated
            if weights[controls].sum() == 0:
                continue
            change = y[:, column[t]] - base
            if x is None:
                expected = np.average(change[controls], weights=weights[controls])
            else:
                design = x[:, column[g]]
                expected = _wls_predict(
                    design[controls], change[controls], weights[controls], design[treated]
                )
            effect = np.average(change[treated] - expected, weights=weights[treated])
            totals[e] = totals.get(e, 0.0) + size * effect
            mass[e] = mass.get(e, 0.0) + size
    return pd.Series({e: totals[e] / mass[e] for e in sorted(totals)})


def _summarise(point: pd.Series, draws: pd.DataFrame, windows: dict[str, tuple[int, int]]) -> dict:
    out = {}
    for name, (first, last) in windows.items():
        cols = [e for e in point.index if first <= e <= last]
        estimate, se = float(point[cols].mean()), float(draws[cols].mean(axis=1).std())
        out[name] = {
            "estimate": estimate,
            "se": se,
            "ci_low": estimate - 1.96 * se,
            "ci_high": estimate + 1.96 * se,
        }
    return out


def _detrend(effects: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Remove a linear pre-campaign drift from each row of event-time effects.

    The drift is a line through the pre-period effects, anchored at zero in the middle of
    the base period, and extrapolated through the campaign weeks. This answers: "what if
    the gap that was opening up before the campaign had simply kept opening?"
    """
    anchor = -(BASE_WEEKS + 1) / 2
    distance = effects.columns.to_numpy(dtype=float) - anchor
    pre = effects.columns < -BASE_WEEKS
    slope = (effects.loc[:, pre] * distance[pre]).sum(axis=1) / (distance[pre] ** 2).sum()
    return effects - np.outer(slope, distance), slope


def estimate(
    y,
    cohort,
    weeks,
    event_window,
    x=None,
    n_boot: int = 300,
    seed: int = SEED,
    summary_windows: dict[str, tuple[int, int]] | None = None,
) -> dict:
    """Point estimates by event time plus bootstrap standard errors and summaries."""
    windows = summary_windows or {}
    point = att_by_event_time(y, cohort, weeks, event_window, x)
    rng = np.random.default_rng(seed)
    draws = pd.DataFrame(
        [
            att_by_event_time(y, cohort, weeks, event_window, x, rng.exponential(size=len(cohort)))
            for _ in range(n_boot)
        ]
    )
    se = draws.std()
    out = {
        "event_time": [int(e) for e in point.index],
        "estimate": point.tolist(),
        "se": se.tolist(),
        "ci_low": (point - 1.96 * se).tolist(),
        "ci_high": (point + 1.96 * se).tolist(),
    } | _summarise(point, draws, windows)

    # Joint test that every pre-period effect is zero (Wald, bootstrap covariance).
    pre = [e for e in point.index if e < -BASE_WEEKS]
    cov = np.cov(draws[pre].to_numpy(), rowvar=False)
    stat = float(point[pre] @ np.linalg.solve(cov, point[pre]))
    out["pretrend_test"] = {"chi2": stat, "df": len(pre), "p_value": float(chi2.sf(stat, len(pre)))}

    detrended_point, slope = _detrend(point.to_frame().T)
    detrended_draws, slope_draws = _detrend(draws)
    out["detrended"] = {
        "drift_per_week": float(slope.iloc[0]),
        "drift_se": float(slope_draws.std()),
    } | _summarise(detrended_point.iloc[0], detrended_draws, windows)
    return out


def load_panel(outcome: str = "spend"):
    """Outcome matrix, cohorts and pre-period covariates from the warehouse."""
    panel = query(f"""
        select w.household_id, w.week, w.{outcome} as y, w.spend, w.trips,
               coalesce(h.first_campaign_week, {NEVER}) as cohort
        from fct_household_week as w
        inner join dim_household as h on w.household_id = h.household_id
        where w.week between {FIRST_FULL_WEEK} and {LAST_WEEK}
          -- households already in a campaign when the data starts have no clean pre-period
          and coalesce(h.first_campaign_week, {NEVER}) > {FIRST_FULL_WEEK} + 1
    """)
    wide = panel.pivot(index="household_id", columns="week", values="y").sort_index()
    cohort = panel.groupby("household_id").cohort.first().reindex(wide.index).to_numpy()
    weeks = wide.columns.to_numpy()

    # Covariates as of each week: mean spend, its square, mean trips and the share of
    # weeks with a trip over the preceding eight weeks.
    spend = panel.pivot(index="household_id", columns="week", values="spend").sort_index()
    trips = panel.pivot(index="household_id", columns="week", values="trips").sort_index()
    lagged = lambda frame: frame.T.rolling(8, min_periods=1).mean().shift(1).T.fillna(0.0)  # noqa: E731
    mean_spend, mean_trips = lagged(spend).to_numpy(), lagged(trips).to_numpy()
    shopped = lagged((trips > 0).astype(float)).to_numpy()
    x = np.stack(
        [np.ones_like(mean_spend), mean_spend, mean_spend**2 / 100, mean_trips, shopped], axis=2
    )
    return wide.to_numpy(), cohort, weeks, x


def run() -> dict:
    y, cohort, weeks, x = load_panel()
    window = (-12, 8)
    summaries = {"during": (0, 4), "after": (5, 8)}
    treated = cohort != NEVER
    result = {
        "unit": "dollars of in-store spend per household per week",
        "n_households": int(len(cohort)),
        "n_never_treated": int((~treated).sum()),
        "n_cohorts": int(len(np.unique(cohort[treated]))),
        "base_weeks": BASE_WEEKS,
        "treated_mean_spend": float(y[treated].mean()),
        # Main specification: not-yet-treated and never-treated households as controls.
        "main": estimate(y, cohort, weeks, window, summary_windows=summaries),
        # Controls restricted to households that do get a campaign later.
        "eventually_treated_only": estimate(
            y[treated], cohort[treated], weeks, window, summary_windows=summaries
        ),
        # Covariate-adjusted version. Kept as a documented failure: see README.
        "covariate_adjusted": estimate(y, cohort, weeks, window, x, summary_windows=summaries),
    }
    save_result("event_study", result)
    return result


if __name__ == "__main__":
    out = run()
    for key in ("main", "eventually_treated_only", "covariate_adjusted"):
        table = pd.DataFrame({k: out[key][k] for k in ("event_time", "estimate", "se")})
        print(key, out[key]["during"], out[key]["after"], out[key]["pretrend_test"])
        print("  detrended", out[key]["detrended"])
        print(table.round(2).T.to_string())
