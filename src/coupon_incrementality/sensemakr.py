"""Sensitivity to unobserved confounding for a linear model (Cinelli & Hazlett, 2020).

"Making Sense of Sensitivity: Extending Omitted Variable Bias", JRSS-B 82(1).

The formulas are implemented directly: the reference Python package (PySensemakr) does
not run on current pandas. tests/test_sensemakr.py checks them against the paper's
published Darfur example.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t


def partial_r2(t_stat: float, dof: int) -> float:
    return t_stat**2 / (t_stat**2 + dof)


def robustness_value(t_stat: float, dof: int, q: float = 1.0, alpha: float | None = None) -> float:
    """Share of residual variance of both treatment and outcome that a confounder must
    explain to cut the estimate by 100*q percent (or, with `alpha`, to make it
    statistically insignificant at that level)."""
    f = q * abs(t_stat) / np.sqrt(dof)
    if alpha is not None:
        f -= student_t.ppf(1 - alpha / 2, dof - 1) / np.sqrt(dof - 1)
    if f <= 0:
        return 0.0
    return float(0.5 * (np.sqrt(f**4 + 4 * f**2) - f**2))


def benchmark_bound(
    estimate: float,
    se: float,
    dof: int,
    r2_benchmark_treatment: float,
    r2_benchmark_outcome: float,
    k: float = 1.0,
) -> dict:
    """Worst-case estimate if a hidden confounder were k times as strong as a benchmark.

    r2_benchmark_treatment : partial R2 of the benchmark covariate with the treatment
    r2_benchmark_outcome   : partial R2 of the benchmark covariate with the outcome
    """
    r2dz = k * r2_benchmark_treatment / (1 - r2_benchmark_treatment)
    r2zx = (
        k
        * r2_benchmark_treatment**2
        / ((1 - k * r2_benchmark_treatment) * (1 - r2_benchmark_treatment))
    )
    if r2dz >= 1 or r2zx >= 1:
        raise ValueError("Implied confounder explains all remaining treatment variation.")
    r2yz = min(
        1.0,
        ((np.sqrt(k) + np.sqrt(r2zx)) / np.sqrt(1 - r2zx)) ** 2
        * r2_benchmark_outcome
        / (1 - r2_benchmark_outcome),
    )
    bias = np.sqrt(r2yz * r2dz / (1 - r2dz)) * se * np.sqrt(dof)
    adjusted = estimate - np.sign(estimate) * bias
    adjusted_se = np.sqrt((1 - r2yz) / (1 - r2dz)) * se * np.sqrt(dof / (dof - 1))
    half_width = student_t.ppf(0.975, dof - 1) * adjusted_se
    return {
        "k": k,
        "confounder_r2_with_treatment_pct": float(100 * r2dz),
        "confounder_r2_with_outcome_pct": float(100 * r2yz),
        "adjusted_estimate": float(adjusted),
        "adjusted_ci_low": float(adjusted - half_width),
        "adjusted_ci_high": float(adjusted + half_width),
    }
