"""Waterfall arithmetic on hand-made numbers."""

import pytest

from coupon_incrementality.decomposition import STEPS, per_coupon_dollar

BASE = {"A": {"n": 100, "coupon_dollars": 0.5}, "BC": {"n": 50, "coupon_dollars": 0.0}}


def part(naive, causal, after=0.0, cannibalisation=0.0, halo=0.0, se=0.0):
    values = dict(
        naive=naive, causal=causal, after=after, cannibalisation=cannibalisation, halo=halo
    )
    return {
        step: {"estimate": values[step], "se": 0.0 if step == "naive" else se} for step in STEPS
    }


def test_single_family():
    parts = {"A": part(naive=100.0, causal=10.0, after=-2.0)}
    out = per_coupon_dollar(parts, BASE, margin=0.25, mail_cost=0.5, families=("A",))
    # cost per recipient = 0.5 coupons + 0.5 mail = $1
    assert out["coupon_spend"] == pytest.approx(100.0)
    assert out["naive"]["estimate"] == pytest.approx(25.0)
    assert out["causal"]["estimate"] == pytest.approx(2.5)
    assert out["targeting_bias"]["estimate"] == pytest.approx(-22.5)
    assert out["true"]["estimate"] == pytest.approx(2.0)
    assert out["share_of_naive_that_is_real_pct"] == pytest.approx(8.0)


def test_families_are_pooled_by_recipients_and_cost():
    parts = {"A": part(100.0, 10.0), "BC": part(4.0, 1.0, cannibalisation=-0.5, halo=2.0)}
    out = per_coupon_dollar(parts, BASE, margin=0.25, mail_cost=0.5)
    cost = 100 * 1.0 + 50 * 0.5
    assert out["coupon_spend"] == pytest.approx(cost)
    assert out["true"]["estimate"] == pytest.approx(0.25 * (100 * 10 + 50 * 2.5) / cost)


def test_interval_combines_component_errors():
    parts = {"A": part(100.0, 10.0, after=-2.0, se=4.0)}
    out = per_coupon_dollar(parts, BASE, margin=0.25, mail_cost=0.5, families=("A",))
    # causal and after each carry se 4 * 0.25 = 1; the two zero-valued steps carry it too
    assert out["true"]["se"] == pytest.approx(2.0)
    assert out["true"]["ci_low"] == pytest.approx(2.0 - 1.96 * 2.0)


def test_higher_mailing_cost_lowers_the_return():
    parts = {"A": part(100.0, 10.0)}
    cheap = per_coupon_dollar(parts, BASE, 0.25, 0.25, families=("A",))["true"]["estimate"]
    dear = per_coupon_dollar(parts, BASE, 0.25, 1.00, families=("A",))["true"]["estimate"]
    assert cheap > dear
