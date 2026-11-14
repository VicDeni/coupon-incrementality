"""The doubly robust engine recovers a known effect when targeting is confounded."""

import numpy as np
import pandas as pd
import pytest

from coupon_incrementality import aipw


@pytest.fixture(scope="module")
def confounded():
    """Heavy spenders are more likely to be targeted, respond more, and - as in the real
    data - would have seen their spending fall back anyway (mean reversion).

    `y` is a change from the household's own baseline, as in the pipeline.
    """
    rng = np.random.default_rng(0)
    n = 4000
    spend = rng.gamma(4, 12, size=n)
    noise = rng.normal(size=(n, 3))
    propensity = 1 / (1 + np.exp(-(spend - 48) / 20))
    d = rng.binomial(1, propensity)
    effect = 0.05 * spend
    y = -0.15 * (spend - 48) + 3 * noise[:, 0] + d * effect + rng.normal(0, 5, size=n)
    frame = pd.DataFrame(
        {"household_id": np.arange(n), "d": d, "y": y, "spend": spend,
         "x1": noise[:, 0], "x2": noise[:, 1], "x3": noise[:, 2]}
    )  # fmt: skip
    return frame, effect


FEATURES = ["spend", "x1", "x2", "x3"]


def test_naive_comparison_is_badly_biased(confounded):
    frame, effect = confounded
    naive = frame[frame.d == 1].y.mean() - frame[frame.d == 0].y.mean()
    # mean reversion among the targeted hides the whole effect
    assert naive < 0 < effect[frame.d == 1].mean()


def test_att_recovers_the_effect_on_the_treated(confounded):
    frame, effect = confounded
    result = aipw.att_repeated(frame, FEATURES, ["y"], n_rep=2)["y"]
    truth = effect[frame.d == 1].mean()
    assert abs(result["estimate"] - truth) < 3 * result["se"]
    assert result["se"] < 1.0


def test_dr_scores_average_to_the_overall_effect(confounded):
    frame, effect = confounded
    fitted = aipw.cross_fit(frame, FEATURES, ["y"], treated_model=True)
    scores = aipw.dr_scores(frame.y, frame.d, fitted.m, fitted.g0_y, fitted.g1_y)
    se = scores.std() / np.sqrt(len(scores))
    assert abs(scores.mean() - effect.mean()) < 3 * se


def test_cross_fit_never_predicts_a_household_from_its_own_fold():
    rng = np.random.default_rng(1)
    n_households = 300
    frame = pd.DataFrame(
        {"household_id": np.repeat(np.arange(n_households), 3), "x": rng.normal(size=900)}
    )
    frame["d"] = rng.binomial(1, 0.5, size=900)
    frame["y"] = frame.x + rng.normal(size=900)
    fitted = aipw.cross_fit(frame, ["x"], ["y"])
    folds_per_household = fitted.fold.groupby(frame.household_id).nunique()
    assert (folds_per_household == 1).all()
    assert fitted.m.between(aipw.CLIP, 1 - aipw.CLIP).all()


def test_cluster_se_matches_plain_se_without_clustering():
    rng = np.random.default_rng(2)
    influence = rng.normal(size=500)
    influence -= influence.mean()
    plain = np.sqrt((influence**2).mean() / 500)
    assert aipw.cluster_se(influence, np.arange(500)) == pytest.approx(plain)


def test_cluster_se_grows_when_rows_are_duplicated_within_clusters():
    rng = np.random.default_rng(3)
    base = rng.normal(size=300)
    base -= base.mean()
    doubled = np.tile(base, 2)
    clusters = np.tile(np.arange(300), 2)
    assert aipw.cluster_se(doubled, clusters) > 1.3 * aipw.cluster_se(doubled, np.arange(600))
