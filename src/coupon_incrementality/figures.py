"""Charts for the README and memo, drawn from the saved result files."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import FIGURES_DIR, PROPENSITY_BOUNDS, load_result  # noqa: E402
from .dml import FRAME_PATH  # noqa: E402

# Palette: validated categorical slots (blue, orange, aqua) and a blue/red diverging pair.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#898781"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
    }
)


def _dollars(value: float) -> str:
    """Dollar amount with a true minus sign, escaped for matplotlib."""
    sign = "\u2212" if value < 0 else ""
    return sign + r"\$" + f"{abs(value):,.2f}"


def _save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _titles(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, pad=30)
    # "$" is escaped so matplotlib does not read a pair of them as maths
    subtitle = subtitle.replace("$", r"\$")
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, va="bottom", color=INK_2, fontsize=9.5)


def waterfall() -> None:
    d = load_result("decomposition")
    program = d["program"]
    steps = [
        ("Dashboard\n(naive lift)", program["naive"]["estimate"], "total"),
        ("Targeting\nbias", program["targeting_bias"]["estimate"], "change"),
        ("Causal lift\nin campaign", program["causal"]["estimate"], "total"),
        ("After the\ncampaign", program["after"]["estimate"], "change"),
        ("Cannibal-\nisation", program["cannibalisation"]["estimate"], "change"),
        ("Halo", program["halo"]["estimate"], "change"),
        ("True\nincremental", program["true"]["estimate"], "total"),
    ]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [3, 4.2], "wspace": 0.22}
    )

    # Left: the first three steps at full scale.
    running = 0.0
    for i, (_, value, kind) in enumerate(steps[:3]):
        bottom = 0.0 if kind == "total" else running + min(value, 0)
        height = value if kind == "total" else abs(value)
        color = NEUTRAL if i == 0 else (RED if kind == "change" else BLUE)
        left.bar(i, height, bottom=bottom, width=0.55, color=color)
        text = f"${value:,.2f}" if kind == "total" else f"−${abs(value):,.2f}"
        left.text(i, bottom + height + 0.9, text, ha="center", fontweight="bold", fontsize=10)
        running = value if kind == "total" else running + value
    left.set_xticks(range(3), [s[0] for s in steps[:3]])
    left.set_ylim(0, program["naive"]["estimate"] * 1.14)
    left.set_ylabel("Gross profit per $1 of coupon spend")
    left.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    left.grid(axis="x", visible=False)
    _titles(left, "From the dashboard number...", "Most of the apparent return is who was chosen")

    # Right: zoom on the causal steps.
    running = 0.0
    for i, (_, value, kind) in enumerate(steps[2:]):
        if kind == "total":
            bottom, height, color = 0.0, value, BLUE
            running = value if i == 0 else running
        else:
            bottom, height = running + min(value, 0), abs(value)
            color = BLUE if value >= 0 else RED
            running += value
        alpha = 1.0 if kind == "total" else 0.55
        right.bar(i, height, bottom=bottom, width=0.55, color=color, alpha=alpha)
        sign = "" if kind == "total" else ("+" if value >= 0 else "−")
        right.text(
            i, bottom + height + 0.12, f"{sign}${abs(value):,.2f}", ha="center",
            fontweight="bold", fontsize=10,
        )  # fmt: skip
    true = program["true"]
    last = len(steps[2:]) - 1
    right.plot([last + 0.36] * 2, [true["ci_low"], true["ci_high"]], color=INK, lw=1.5)
    interval = f"95% CI\n{_dollars(true['ci_low'])} to {_dollars(true['ci_high'])}"
    right.text(last + 0.44, true["ci_high"], interval, va="top", fontsize=8.5, color=INK_2)
    right.axhline(1, color=INK_2, lw=1)
    right.text(last + 0.44, 1.08, "break-even\n(\\$1)", fontsize=8.5, color=INK_2, va="bottom")
    right.set_xticks(range(len(steps[2:])), [s[0] for s in steps[2:]])
    right.set_xlim(-0.5, last + 1.25)
    right.set_ylim(min(0, true["ci_low"]) - 0.3, max(true["ci_high"], running) * 1.08)
    right.axhline(0, color=AXIS, lw=1)
    right.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    right.grid(axis="x", visible=False)
    _titles(
        right,
        "...to true incremental profit",
        "Zoomed in: what is left after removing targeting bias",
    )
    _save(fig, "waterfall")


def designs() -> None:
    """Every design's estimate of the lift, as a percentage of recipients' spend."""
    dml, es, td, bc, naive = (
        load_result(n) for n in ("dml", "event_study", "triple_diff", "dml_bc", "naive")
    )
    base_a = dml["recipient_mean_spend"]

    def pct(block: dict, base: float) -> list[float]:
        return [100 * block[k] / base for k in ("estimate", "ci_low", "ci_high")]

    bc_total = sum(bc["recipient_weekly_spend"].values())
    bc_weekly = [bc["during"][g]["weekly"] for g in ("covered", "sibling", "rest")]
    bc_est = sum(b["estimate"] for b in bc_weekly)
    bc_se = float(np.sqrt(sum(b["se"] ** 2 for b in bc_weekly)))
    covered = td["main"]["covered_during"]
    detrended = es["main"]["detrended"]["during"]
    covered_pct = [covered[k] for k in ("lift_pct", "lift_pct_low", "lift_pct_high")]
    bc_pct = [100 * v / bc_total for v in (bc_est, bc_est - 1.96 * bc_se, bc_est + 1.96 * bc_se)]
    rows = [
        ("Type A · double ML, effect on recipients", *pct(dml["during"]["att"], base_a), BLUE),
        ("Type A · double ML, partially linear", *pct(dml["during"]["plr"], base_a), BLUE),
        ("First campaign · staggered event study", *pct(es["main"]["during"], base_a), ORANGE),
        ("First campaign · event study, drift removed", *pct(detrended, base_a), ORANGE),
        ("Type B/C · triple difference (covered products)", *covered_pct, AQUA),
        ("Type B/C · double ML (whole basket)", *bc_pct, AQUA),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 3.9))
    for i, (_, est, low, high, color) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([low, high], [y, y], color=color, lw=2, solid_capstyle="round")
        ax.plot(est, y, "o", color=color, ms=9, mec=SURFACE, mew=2)
        ax.text(high + 0.6, y, f"{est:+.1f}%", va="center", fontsize=9.5, fontweight="bold")
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(
        "Lift in recipients' spend while the campaign ran (point estimate and 95% interval)"
    )
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%".replace("-", "\u2212"))
    ax.grid(axis="y", visible=False)
    naive_a = naive["comparison"]["A"]["recipient_lift_pct"]
    naive_bc = td["naive"]["lift_pct"]
    _titles(
        ax,
        "Every causal design puts the lift in single digits",
        f"The naive comparisons say +{naive_a:.0f}% (Type A, whole basket) and "
        f"+{naive_bc:.0f}% (Type B/C, covered products)",
    )
    _save(fig, "designs")


def overlap() -> None:
    frame = pd.read_parquet(FRAME_PATH)
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for d, label, color in [(0, "Non-recipients", ORANGE), (1, "Recipients", BLUE)]:
        ax.hist(
            frame[frame.d == d].propensity,
            bins=bins,
            histtype="stepfilled",
            alpha=0.35,
            color=color,
        )
        ax.hist(
            frame[frame.d == d].propensity,
            bins=bins,
            histtype="step",
            lw=2,
            color=color,
            label=label,
        )
    for bound in PROPENSITY_BOUNDS:
        ax.axvline(bound, color=INK_2, lw=1)
    top = ax.get_ylim()[1]
    ax.text(
        PROPENSITY_BOUNDS[0] + 0.01,
        top * 0.97,
        "trimmed\nbelow 0.05",
        fontsize=8.5,
        color=INK_2,
        va="top",
    )
    ax.text(
        PROPENSITY_BOUNDS[1] - 0.01,
        top * 0.97,
        "trimmed\nabove 0.95",
        fontsize=8.5,
        color=INK_2,
        va="top",
        ha="right",
    )
    ax.set_xlabel("Estimated probability of receiving the Type A campaign (out-of-fold)")
    ax.set_ylabel("Household-campaigns")
    ax.legend(loc="center", ncol=2)
    ax.grid(axis="x", visible=False)
    share = (~frame.in_support).mean()
    _titles(
        ax,
        "Who has a comparable counterpart?",
        f"{share:.0%} of household-campaigns fall outside the band and are dropped, "
        "mostly light shoppers",
    )
    _save(fig, "overlap")


def event_studies() -> None:
    es, td = load_result("event_study")["main"], load_result("triple_diff")["event_study"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"wspace": 0.25})

    weeks, est = np.array(es["event_time"]), np.array(es["estimate"])
    low, high = np.array(es["ci_low"]), np.array(es["ci_high"])
    for part in (weeks < 0, weeks >= 0):
        a.fill_between(weeks[part], low[part], high[part], color=ORANGE, alpha=0.12, lw=0)
        a.plot(weeks[part], est[part], "-o", color=ORANGE, lw=2, ms=6, mec=SURFACE, mew=1.5)
    a.axhline(0, color=AXIS, lw=1)
    a.axvspan(-4.5, -0.5, color=GRID, alpha=0.5, lw=0)
    a.text(
        -2.5,
        a.get_ylim()[0] * 0.92,
        "base\nperiod",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=INK_2,
    )
    a.set_xticks(range(-12, 9, 4))
    a.set_xlabel("Weeks since the household's first campaign started")
    a.set_ylabel("Effect on weekly spend")
    a.yaxis.set_major_formatter(lambda v, _: f"{'−' if v < 0 else ''}${abs(v):.0f}")
    p = es["pretrend_test"]["p_value"]
    _titles(
        a,
        "Household level: staggered event study",
        f"Pre-period joint test p = {p:.2f}, but a visible downward drift",
    )

    weeks = np.array(td["rel_week"])
    est = 100 * np.expm1(np.array(td["coef"]))
    low, high = 100 * np.expm1(np.array(td["ci_low"])), 100 * np.expm1(np.array(td["ci_high"]))
    for part in (weeks < 0, weeks == 0, weeks > 0):
        b.fill_between(weeks[part], low[part], high[part], color=AQUA, alpha=0.12, lw=0)
        b.plot(weeks[part], est[part], "-o", color=AQUA, lw=2, ms=6, mec=SURFACE, mew=1.5)
    b.errorbar(
        0,
        est[weeks == 0],
        yerr=[est[weeks == 0] - low[weeks == 0], high[weeks == 0] - est[weeks == 0]],
        color=AQUA,
        lw=2,
    )
    b.axhline(0, color=AXIS, lw=1)
    b.set_xticks(
        [-8, -6, -4, -2, 0, 2, 4, 6, 8], ["−8", "−6", "−4", "−2", "during", "+2", "+4", "+6", "+8"]
    )
    b.set_xlabel("Weeks before the campaign starts / after it ends")
    b.set_ylabel("Effect on covered-product spend")
    b.yaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%".replace("-", "\u2212"))
    p = td["pretrend_test"]["p_value"]
    _titles(
        b,
        "Product level: triple difference",
        f"Flat before the campaign (joint test p = {p:.2f}); no dip after it",
    )
    _save(fig, "event_study")


def gates() -> None:
    h = load_result("heterogeneity")
    g = h["gates"]
    x, effect, se = np.array(g["group"]), np.array(g["effect"]), np.array(g["se"])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(x, effect, width=0.5, color=BLUE)
    ax.errorbar(x, effect, yerr=1.96 * se, fmt="none", ecolor=INK, lw=1.2, capsize=3)
    for xi, e in zip(x, effect, strict=True):
        ax.text(xi + 0.3, e, f"${e:.2f}", va="center", fontsize=9.5, fontweight="bold")
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(x, ["Bottom fifth", "2nd", "3rd", "4th", "Top fifth"])
    ax.set_xlabel("Households sorted by the causal forest's predicted response (out of sample)")
    ax.set_ylabel("Measured effect on weekly spend")
    ax.yaxis.set_major_formatter(lambda v, _: f"{'−' if v < 0 else ''}${abs(v):.0f}")
    ax.grid(axis="x", visible=False)
    _titles(
        ax,
        "Do predicted responders actually respond more?",
        f"Top minus bottom fifth: +${g['top_minus_bottom']:.2f} per week "
        f"(p = {g['top_minus_bottom_p_value']:.2f}), not distinguishable from noise",
    )
    _save(fig, "gates")


def policies() -> None:
    p = load_result("policy")
    base = p["base"]
    order = [
        "Retailer's actual targeting",
        "Random",
        "Causal forest",
        "Top spenders",
        "Mail everyone",
    ]
    fig, ax = plt.subplots(figsize=(9, 3.7))
    for i, name in enumerate(order):
        profit = base[name]["profit"]
        color = NEUTRAL if name in ("Retailer's actual targeting", "Mail everyone") else BLUE
        ax.barh(i, profit["estimate"], height=0.5, color=color)
        ax.plot([profit["ci_low"], profit["ci_high"]], [i, i], color=INK, lw=1.2)
        ax.text(
            max(profit["ci_high"], 0) + 700, i, f"${profit['estimate']:,.0f}", va="center",
            fontsize=9.5, fontweight="bold",
        )  # fmt: skip
    labels = [f"{n}\n({base[n]['mailers']:,.0f} mailers)" for n in order]
    ax.set_yticks(range(len(order)), labels)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlabel("Incremental profit vs. mailing nobody (doubly robust estimate, 95% interval)")
    ax.xaxis.set_major_formatter(lambda v, _: f"{'−' if v < 0 else ''}${abs(v) / 1000:,.0f}K")
    ax.grid(axis="y", visible=False)
    _titles(
        ax,
        "Targeting policies at the retailer's own budget",
        "Type A campaigns, held-out households. The intervals overlap: no rule is a proven winner",
    )
    _save(fig, "policy")


def run() -> None:
    for draw in (waterfall, designs, overlap, event_studies, gates, policies):
        draw()
    print(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    run()
