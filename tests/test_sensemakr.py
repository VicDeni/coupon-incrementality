"""Sensitivity formulas reproduce the worked example in Cinelli & Hazlett (2020).

Darfur example, Table 1 of the paper: estimate 0.097, standard error 0.023,
t-value 4.18, 783 degrees of freedom -> partial R2 2.2%, RV 13.9%, RV(alpha=0.05) 7.6%.
"""

import pytest

from coupon_incrementality import sensemakr

T_STAT, DOF = 4.18, 783


def test_partial_r2():
    assert 100 * sensemakr.partial_r2(T_STAT, DOF) == pytest.approx(2.2, abs=0.05)


def test_robustness_value():
    assert 100 * sensemakr.robustness_value(T_STAT, DOF) == pytest.approx(13.9, abs=0.1)


def test_robustness_value_for_significance():
    assert 100 * sensemakr.robustness_value(T_STAT, DOF, alpha=0.05) == pytest.approx(7.6, abs=0.1)


def test_insignificant_estimate_has_zero_robustness():
    assert sensemakr.robustness_value(1.0, DOF, alpha=0.05) == 0.0


def test_benchmark_bound_matches_the_paper():
    # Paper, Table 2 ("1x female"): R2 with treatment 0.9%, with outcome 12.5%,
    # adjusted estimate 0.075. Benchmark partial R2s: 0.91% (treatment), 10.9% (outcome).
    bound = sensemakr.benchmark_bound(0.097, 0.023, DOF, 0.0091, 0.109, k=1)
    assert bound["confounder_r2_with_treatment_pct"] == pytest.approx(0.9, abs=0.05)
    assert bound["confounder_r2_with_outcome_pct"] == pytest.approx(12.5, abs=0.3)
    assert bound["adjusted_estimate"] == pytest.approx(0.075, abs=0.002)


def test_stronger_confounders_move_the_estimate_further():
    weak = sensemakr.benchmark_bound(2.0, 0.5, 1000, 0.02, 0.05, k=1)
    strong = sensemakr.benchmark_bound(2.0, 0.5, 1000, 0.02, 0.05, k=3)
    assert strong["adjusted_estimate"] < weak["adjusted_estimate"] < 2.0
