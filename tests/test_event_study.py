"""The staggered event-study estimator recovers a known effect on simulated data."""

import numpy as np
import pytest

from coupon_incrementality.event_study import BASE_WEEKS, NEVER, att_by_event_time, estimate


def simulate(n=3000, n_weeks=40, seed=0, drift=0.0):
    """Staggered adoption with an effect that grows with time since treatment.

    Later cohorts respond more strongly, the textbook case in which a two-way fixed
    effects regression is biased but Callaway-Sant'Anna is not.
    """
    rng = np.random.default_rng(seed)
    cohort = rng.choice([12, 18, 24, 30, NEVER], size=n, p=[0.2, 0.2, 0.2, 0.2, 0.2])
    weeks = np.arange(1, n_weeks + 1)
    household = rng.normal(50, 15, size=n)[:, None]
    season = 5 * np.sin(weeks / 6)[None, :]
    since = weeks[None, :] - cohort[:, None]
    strength = np.where(cohort == NEVER, 0.0, cohort / 12)[:, None]
    effect = np.where(since >= 0, strength * (2 + 0.5 * since), 0.0)
    trend = drift * (cohort != NEVER)[:, None] * weeks[None, :]
    y = household + season + trend + effect + rng.normal(0, 4, size=(n, n_weeks))
    return y, cohort, weeks


def true_effect(cohort, event_time):
    cohorts, sizes = np.unique(cohort[cohort != NEVER], return_counts=True)
    return np.average(cohorts / 12 * (2 + 0.5 * event_time), weights=sizes)


def test_recovers_dynamic_effects():
    y, cohort, weeks = simulate()
    estimates = att_by_event_time(y, cohort, weeks, (-8, 6))
    for event_time in range(0, 7):
        assert estimates[event_time] == pytest.approx(true_effect(cohort, event_time), abs=0.6)


def test_pre_period_effects_are_zero_without_drift():
    y, cohort, weeks = simulate()
    estimates = att_by_event_time(y, cohort, weeks, (-8, 6))
    pre = estimates[estimates.index < -BASE_WEEKS]
    assert len(pre) == 8 - BASE_WEEKS
    assert np.abs(pre).max() < 0.6


def test_base_period_is_excluded():
    y, cohort, weeks = simulate()
    estimates = att_by_event_time(y, cohort, weeks, (-8, 6))
    assert not set(range(-BASE_WEEKS, 0)) & set(estimates.index)


def test_covariate_adjustment_runs_and_agrees_when_covariates_are_irrelevant():
    y, cohort, weeks = simulate()
    rng = np.random.default_rng(1)
    x = np.stack([np.ones_like(y), rng.normal(size=y.shape)], axis=2)
    plain = att_by_event_time(y, cohort, weeks, (0, 4))
    adjusted = att_by_event_time(y, cohort, weeks, (0, 4), x=x)
    assert np.allclose(plain, adjusted, atol=0.3)


def test_detrending_removes_a_linear_drift():
    y, cohort, weeks = simulate(drift=-0.3)
    out = estimate(y, cohort, weeks, (-8, 4), n_boot=40, summary_windows={"during": (0, 4)})
    truth = np.mean([true_effect(cohort, e) for e in range(5)])
    # The raw estimate is pulled down by the drift; the detrended one is not. (Controls
    # that are treated later share the drift, so the measured drift is smaller than 0.3.)
    raw, detrended = out["during"]["estimate"], out["detrended"]["during"]["estimate"]
    assert raw < truth - 0.4
    assert abs(detrended - truth) < abs(raw - truth)
    assert detrended == pytest.approx(truth, abs=0.6)
    assert out["detrended"]["drift_per_week"] < -0.05


def test_bootstrap_interval_covers_the_truth():
    y, cohort, weeks = simulate(n=1500, seed=3)
    out = estimate(y, cohort, weeks, (-8, 4), n_boot=60, summary_windows={"during": (0, 4)})
    truth = np.mean([true_effect(cohort, e) for e in range(5)])
    assert out["during"]["ci_low"] < truth < out["during"]["ci_high"]
    assert out["pretrend_test"]["p_value"] > 0.01
