"""Phase 4 - where the money really goes.

Assembles the waterfall from the dashboard number to true incremental gross profit per
$1 of coupon spend:

    naive lift
      - targeting bias          -> causal lift during the campaign
      +/- after-campaign effect    (pull-forward shows up here as a negative)
      - cannibalisation            (sibling products in the same category)
      + halo                       (rest of the store)
      = true incremental sales  x gross margin  / coupon spend

Sources, by campaign family:

                      Type A (personalised)              Type B/C (same coupons for all)
    causal lift       double ML, whole basket            triple difference, covered products
    after-campaign    double ML, 4 weeks after           triple difference, 8 weeks after
    cannibalisation   already inside the basket effect   triple difference, sibling products
    halo              already inside the basket effect   double ML, rest of store

Type A's 16 coupons per household are not recorded, so its effect is measured on the
whole basket, which already nets out switching between products and includes any halo.

"Coupon spend" is the face value of redeemed campaign coupons plus an assumed cost of
printing and mailing each campaign to each household.
"""

from __future__ import annotations

import numpy as np

from .config import (
    DISTRIBUTION_COST,
    MARGIN_BASE,
    MARGIN_HIGH,
    MARGIN_LOW,
    TYPE_A_CAMPAIGNS,
    load_result,
    save_result,
)
from .db import query

STEPS = ("naive", "causal", "after", "cannibalisation", "halo")


def _est(estimate: float, se: float = 0.0) -> dict:
    return {"estimate": float(estimate), "se": float(se)}


def _from_ci(block: dict) -> dict:
    return _est(block["dollars_per_recipient"], (block["ci_high"] - block["ci_low"]) / 3.92)


def recipients_and_costs() -> dict:
    """Recipient-campaigns and redeemed coupon dollars per recipient, by family."""
    frame = query(f"""
        select
            case when campaign_type = 'A' then 'A' else 'BC' end as family,
            count(*) as n,
            avg(coupon_value) as coupon_dollars
        from fct_household_campaign
        where campaign_id in {TYPE_A_CAMPAIGNS}
           or campaign_id in (select distinct campaign_id from fct_household_group_week)
        group by 1
    """).set_index("family")
    return {
        f: {"n": int(frame.n[f]), "coupon_dollars": float(frame.coupon_dollars[f])}
        for f in frame.index
    }


def components() -> dict:
    """Incremental sales dollars per recipient household per campaign, by family and step."""
    dml, td, bc = load_result("dml"), load_result("triple_diff"), load_result("dml_bc")
    naive_a = load_result("naive")["comparison"]["A"]
    weeks_a = dml["mean_campaign_weeks"]
    att, post = dml["during"]["att"], dml["post"]
    return {
        "A": {
            "naive": _est(
                (naive_a["recipients_weekly_spend"] - naive_a["non_recipients_weekly_spend"])
                * weeks_a
            ),
            "causal": _est(att["estimate"] * weeks_a, att["se"] * weeks_a),
            "after": _est(
                post["att"]["estimate"] * post["weeks"], post["att"]["se"] * post["weeks"]
            ),
            "cannibalisation": _est(0.0),
            "halo": _est(0.0),
        },
        "BC": {
            "naive": _est(td["naive"]["difference"] * td["dollars"]["weeks_during"]),
            "causal": _from_ci(td["dollars"]["covered_during"]),
            "after": _from_ci(td["dollars"]["covered_post"]),
            "cannibalisation": _from_ci(td["dollars"]["sibling_during"]),
            "halo": _est(bc["during"]["rest"]["estimate"], bc["during"]["rest"]["se"]),
        },
    }


def per_coupon_dollar(
    parts: dict, base: dict, margin: float, mail_cost: float, families=("A", "BC")
) -> dict:
    """Gross profit per $1 of coupon spend for each waterfall step, pooled over families.

    The confidence interval of the total treats the component estimates as independent,
    which is an approximation (they share households).
    """
    cost = sum(base[f]["n"] * (base[f]["coupon_dollars"] + mail_cost) for f in families)
    scale = {f: base[f]["n"] * margin / cost for f in families}
    out = {
        step: {
            "estimate": sum(scale[f] * parts[f][step]["estimate"] for f in families),
            "se": float(np.sqrt(sum((scale[f] * parts[f][step]["se"]) ** 2 for f in families))),
        }
        for step in STEPS
    }
    out["targeting_bias"] = {"estimate": out["causal"]["estimate"] - out["naive"]["estimate"]}
    true = sum(out[s]["estimate"] for s in STEPS[1:])
    se = float(np.sqrt(sum(out[s]["se"] ** 2 for s in STEPS[1:])))
    out["true"] = {
        "estimate": true,
        "se": se,
        "ci_low": true - 1.96 * se,
        "ci_high": true + 1.96 * se,
    }
    out["share_of_naive_that_is_real_pct"] = 100 * true / out["naive"]["estimate"]
    out["coupon_spend"] = cost
    return out


def run() -> dict:
    base, parts = recipients_and_costs(), components()
    dml, es = load_result("dml"), load_result("event_study")
    weeks_a = dml["mean_campaign_weeks"]

    def with_type_a_effect(weekly: dict) -> dict:
        """Type A alone, swapping in a different design's weekly effect."""
        swapped = {
            "A": parts["A"] | {"causal": _est(weekly["estimate"] * weeks_a, weekly["se"] * weeks_a)}
        }
        return per_coupon_dollar(swapped, base, MARGIN_BASE, DISTRIBUTION_COST, families=("A",))[
            "true"
        ]

    no_halo = {"A": parts["A"], "BC": parts["BC"] | {"halo": _est(0.0)}}
    result = {
        "unit": "gross profit dollars per $1 of coupon spend, unless stated",
        "assumptions": {
            "margin": MARGIN_BASE,
            "mail_cost_per_household_campaign": DISTRIBUTION_COST,
        },
        "recipients_and_costs": base,
        "sales_per_recipient_campaign": parts,
        "program": per_coupon_dollar(parts, base, MARGIN_BASE, DISTRIBUTION_COST),
        "type_a": per_coupon_dollar(parts, base, MARGIN_BASE, DISTRIBUTION_COST, families=("A",)),
        "type_bc": per_coupon_dollar(parts, base, MARGIN_BASE, DISTRIBUTION_COST, families=("BC",)),
        "type_bc_without_halo": per_coupon_dollar(
            no_halo, base, MARGIN_BASE, DISTRIBUTION_COST, families=("BC",)
        ),
        # How the answer moves with the two assumptions that are not in the data.
        "sensitivity": [
            {
                "margin": margin,
                "mail_cost": mail,
                "true": per_coupon_dollar(parts, base, margin, mail)["true"]["estimate"],
                "naive": per_coupon_dollar(parts, base, margin, mail)["naive"]["estimate"],
            }
            for margin in (MARGIN_LOW, MARGIN_BASE, MARGIN_HIGH)
            for mail in (0.25, 0.50, 1.00)
        ],
        # Type A under each household-level design (same after-campaign term throughout).
        "type_a_by_design": {
            "Double ML, effect on recipients (primary)": with_type_a_effect(dml["during"]["att"]),
            "Double ML, partially linear": with_type_a_effect(dml["during"]["plr"]),
            "Event study": with_type_a_effect(es["main"]["during"]),
            "Event study, pre-campaign drift removed": with_type_a_effect(
                es["main"]["detrended"]["during"]
            ),
        },
    }
    save_result("decomposition", result)
    return result


if __name__ == "__main__":
    import json

    out = run()
    for key in (
        "recipients_and_costs",
        "program",
        "type_a",
        "type_bc",
        "type_bc_without_halo",
        "type_a_by_design",
    ):
        print(key, json.dumps(out[key], indent=1, default=float))
