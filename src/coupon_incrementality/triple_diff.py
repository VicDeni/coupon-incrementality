"""Phase 3b - triple difference for Type B and Type C campaigns.

Every participant in a Type B/C campaign received every coupon, so we know exactly which
products each household held a coupon for. That allows three contrasts at once:

    recipients vs. non-recipients  x  covered vs. other products  x  campaign vs. other weeks

The model is a Poisson regression (PPML) with three sets of fixed effects, estimated on a
stack of one block per campaign:

    household x week   - anything that moves a household's whole basket that week
    group x week       - price changes, seasonality and promotions common to a product group
    household x group  - a household's standing taste for those products

Why Poisson rather than OLS: recipients spend three times as much as non-recipients, and
covered products are about 2% of a basket. A payday or a holiday moves both groups and
all products by a similar *percentage*, not a similar dollar amount, so the fixed effects
only do their job in a multiplicative model. (The OLS version is reported in the results
as a cautionary example; its "effects" are artefacts of scale.)

Coefficients are log points: exp(b) - 1 is the percentage lift. Standard errors are
clustered by household.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy.stats import chi2

from .config import save_result
from .db import query

GROUPS = ("covered", "sibling", "placebo")  # "other" is the reference group
CONTROLS = "display_share + mailer_share"
FIXED_EFFECTS = "hw + gw + hg"


def load_panel(where: str = "rel_week >= -8") -> pd.DataFrame:
    panel = query(f"""
        select
            campaign_id, household_id, week, product_group, period,
            cast(rel_week as integer) as rel_week,
            cast(is_recipient as integer) as rec,
            n_overlapping_campaigns, spend, display_share, mailer_share
        from fct_household_group_week
        where {where}
    """)
    for column in ("product_group", "period"):
        panel[column] = panel[column].astype("category")
    # integer ids for the fixed effects
    ids = {
        "hw": ["campaign_id", "household_id", "week"],
        "gw": ["campaign_id", "product_group", "week"],
        "hg": ["campaign_id", "household_id", "product_group"],
        "hc": ["campaign_id", "household_id"],
        "wc": ["campaign_id", "week"],
    }
    for name, keys in ids.items():
        panel[name] = panel.groupby(keys, observed=True).ngroup().astype("int32")
    return panel


def informative(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop cells a Poisson model cannot learn from (fixed-effect groups that are all zero).

    The estimator would drop them anyway; doing it first keeps memory in check.
    """
    size = -1
    while size != len(panel):
        size = len(panel)
        for key in ("hw", "hg"):
            panel = panel[panel.groupby(key).spend.transform("sum") > 0]
    return panel


def add_terms(panel: pd.DataFrame, periods=("during", "post")) -> list[str]:
    """Add recipient x group x period indicators in place; return their names."""
    names = []
    for group in GROUPS:
        for period in periods:
            name = f"{group}_{period}"
            panel[name] = (
                (panel.rec == 1) & (panel.product_group == group) & (panel.period == period)
            ).astype(float)
            names.append(name)
    return names


def _tidy(fit, terms: list[str]) -> dict:
    table = fit.tidy()
    out = {}
    for term in terms:
        row = table.loc[term]
        out[term] = {
            "coef": float(row["Estimate"]),
            "se": float(row["Std. Error"]),
            "p_value": float(row["Pr(>|t|)"]),
            "lift_pct": float(100 * np.expm1(row["Estimate"])),
            "lift_pct_low": float(100 * np.expm1(row["2.5%"])),
            "lift_pct_high": float(100 * np.expm1(row["97.5%"])),
        }
    return out


def fit_ppml(panel: pd.DataFrame, terms: list[str]) -> dict:
    fit = pf.fepois(
        f"spend ~ {' + '.join(terms)} + {CONTROLS} | {FIXED_EFFECTS}",
        informative(panel),
        vcov={"CRV1": "household_id"},
    )
    return _tidy(fit, terms + ["display_share", "mailer_share"]) | {"n_obs": int(fit._N)}


def _added(observed: float, lift_pct: float) -> float:
    """Dollars a lift of `lift_pct` percent added to an observed (post-lift) amount."""
    return float(observed * (1 - 1 / (1 + lift_pct / 100)))


def to_dollars(panel: pd.DataFrame, estimates: dict) -> dict:
    """Convert percentage lifts into dollars per recipient.

    If recipients' observed spend is S and the coupon multiplied it by exp(b), then spend
    without the coupon would have been S / exp(b), and the coupon added S * (1 - exp(-b)).
    """
    recipients = panel[panel.rec == 1]
    households = recipients[["campaign_id", "household_id"]].drop_duplicates()
    out = {"n_recipient_campaigns": int(len(households))}
    for period in ("during", "post"):
        rows = recipients[recipients.period == period]
        weeks_per_recipient = rows[rows.product_group == "covered"].shape[0] / len(households)
        out[f"weeks_{period}"] = float(weeks_per_recipient)
        for group in GROUPS:
            est = estimates[f"{group}_{period}"]
            observed = rows[rows.product_group == group].spend.mean() * weeks_per_recipient
            out[f"{group}_{period}"] = {
                "observed_weekly_spend": float(observed / weeks_per_recipient),
                "dollars_per_recipient": _added(observed, est["lift_pct"]),
                "ci_low": _added(observed, est["lift_pct_low"]),
                "ci_high": _added(observed, est["lift_pct_high"]),
            }
    return out


def event_study(panel: pd.DataFrame) -> dict:
    """Week-by-week effect on covered products; the week before the start is the baseline.

    Estimated on covered and reference products only, which halves the memory needed.
    """
    panel = informative(panel[panel.product_group.isin(["covered", "other"])]).copy()
    weeks = [w for w in range(-8, 9) if w != -1]
    terms = []
    for week in weeks:
        name = f"covered_{'m' if week < 0 else 'p'}{abs(week)}"
        panel[name] = (
            (panel.rec == 1) & (panel.product_group == "covered") & (panel.rel_week == week)
        ).astype(float)
        terms.append(name)
    fit = pf.fepois(
        f"spend ~ {' + '.join(terms)} + {CONTROLS} | {FIXED_EFFECTS}",
        panel,
        vcov={"CRV1": "household_id"},
    )
    table = fit.tidy().loc[terms]
    # Joint test that all pre-campaign coefficients are zero.
    names = list(fit._coefnames)
    pre = [names.index(t) for t, w in zip(terms, weeks, strict=True) if w < 0]
    beta, cov = fit.coef().to_numpy()[pre], np.asarray(fit._vcov)[np.ix_(pre, pre)]
    stat = float(beta @ np.linalg.solve(cov, beta))
    return {
        "rel_week": weeks,
        "coef": table["Estimate"].tolist(),
        "se": table["Std. Error"].tolist(),
        "ci_low": table["2.5%"].tolist(),
        "ci_high": table["97.5%"].tolist(),
        "pretrend_test": {"chi2": stat, "df": len(pre), "p_value": float(chi2.sf(stat, len(pre)))},
    }


def naive(panel: pd.DataFrame) -> dict:
    """What a dashboard would report: recipients vs. non-recipients, campaign weeks only."""
    during = panel[(panel.period == "during") & (panel.product_group == "covered")]
    by_campaign = during.groupby(["campaign_id", "rec"]).spend.mean().unstack()
    n_recipients = during[during.rec == 1].groupby("campaign_id").household_id.nunique()
    difference = np.average(by_campaign[1] - by_campaign[0], weights=n_recipients)
    recipients = np.average(by_campaign[1], weights=n_recipients)
    return {
        "recipients_weekly_covered_spend": float(recipients),
        "non_recipients_weekly_covered_spend": float(recipients - difference),
        "difference": float(difference),
        "lift_pct": float(100 * difference / (recipients - difference)),
    }


def halo_did(panel: pd.DataFrame) -> dict:
    """Difference-in-differences on rest-of-store spend (the halo).

    The household x week fixed effect of the triple difference absorbs any store-wide
    effect, so halo needs a weaker two-way design. Its pre-trend test fails (see the
    README), so this estimate is reported but not used.
    """
    rest = panel[panel.product_group.isin(["other", "placebo"])]
    rest = rest.groupby(
        ["campaign_id", "household_id", "rel_week", "period", "rec", "hc", "wc"],
        as_index=False,
        observed=True,
    ).spend.sum()
    rest["during"] = ((rest.rec == 1) & (rest.period == "during")).astype(float)
    rest["post"] = ((rest.rec == 1) & (rest.period == "post")).astype(float)
    pooled = pf.fepois("spend ~ during + post | hc + wc", rest, vcov={"CRV1": "household_id"})

    weeks = [w for w in range(-8, 9) if w != -1]
    terms = []
    for week in weeks:
        name = f"rest_{'m' if week < 0 else 'p'}{abs(week)}"
        rest[name] = ((rest.rec == 1) & (rest.rel_week == week)).astype(float)
        terms.append(name)
    dynamic = pf.fepois(
        f"spend ~ {' + '.join(terms)} | hc + wc", rest, vcov={"CRV1": "household_id"}
    )
    names = list(dynamic._coefnames)
    pre = [names.index(t) for t, w in zip(terms, weeks, strict=True) if w < 0]
    beta, cov = dynamic.coef().to_numpy()[pre], np.asarray(dynamic._vcov)[np.ix_(pre, pre)]
    stat = float(beta @ np.linalg.solve(cov, beta))
    table = dynamic.tidy().loc[terms]
    return {
        "pooled": _tidy(pooled, ["during", "post"]),
        "rel_week": weeks,
        "coef": table["Estimate"].tolist(),
        "se": table["Std. Error"].tolist(),
        "pretrend_test": {"chi2": stat, "df": len(pre), "p_value": float(chi2.sf(stat, len(pre)))},
    }


def ols_levels(panel: pd.DataFrame, terms: list[str]) -> dict:
    fit = pf.feols(
        f"spend ~ {' + '.join(terms)} + {CONTROLS} | {FIXED_EFFECTS}",
        panel,
        vcov={"CRV1": "household_id"},
    )
    table = fit.tidy()
    return {
        t: {"coef": float(table.loc[t, "Estimate"]), "se": float(table.loc[t, "Std. Error"])}
        for t in terms
    }


def run() -> dict:
    panel = load_panel()
    terms = add_terms(panel)
    main = fit_ppml(panel, terms)
    result = {
        "unit": "log points; lift_pct = 100 * (exp(coef) - 1)",
        "n_campaigns": int(panel.campaign_id.nunique()),
        "campaigns": sorted(int(c) for c in panel.campaign_id.unique()),
        "n_rows": int(len(panel)),
        "naive": naive(panel),
        "main": main,
        "dollars": to_dollars(panel, main),
        "ols_levels": ols_levels(panel, terms),
        "halo_did": halo_did(panel),
    }
    result["event_study"] = event_study(panel.drop(columns=terms))
    save_result("triple_diff", result)
    return result


if __name__ == "__main__":
    out = run()
    print(pd.DataFrame(out["main"]).T.round(3))
    print(out["naive"], out["dollars"], out["event_study"]["pretrend_test"], sep="\n")
    print(out["halo_did"]["pooled"], out["halo_did"]["pretrend_test"])
