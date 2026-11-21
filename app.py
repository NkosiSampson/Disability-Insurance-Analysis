"""
Disability policy explorer, Python / Streamlit version.

Run it (in the same folder as disability_model.py):
    pip install streamlit altair numpy pandas
    streamlit run app.py
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from disability_model import (BENEFIT_PERIODS, ELIM_OPTIONS, Person, Policy, baseline, default_income_path,
                              exceed, run_policy, simulate, years_below_floor, Money)

st.set_page_config(page_title="Disability policy explorer", page_icon="🛡️", layout="wide")

NAMES = ["No policy", "Policy A", "Policy B", "Policy C"]
COLORS = ["#B3362B", "#16707A", "#6A4DB0", "#A17010"]


def money(x):
    a = abs(x)
    s = f"{a/1e6:.1f}M" if a >= 1e6 else f"{a/1e3:.0f}k" if a >= 1e3 else f"{a:.0f}"
    return ("−" if x < 0 else "") + "$" + s


def per100(q):
    v = q * 100
    return "0" if v <= 0 else "fewer than 1" if v < 1 else f"{round(v * 2) / 2:g}" if v < 10 else f"{v:.0f}"


def md(text):
    """Escape dollar signs so Streamlit doesn't read them as math."""
    return text.replace("\\$", "$").replace("$", "\\$")


def yrs(x):
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}" if x >= 1 else f"{x:.2f}"


@st.cache_data(show_spinner="Simulating 50,000 working lives…")
def cached_sim(age, sex, income, occ, smoker, bmi, chronic, hobby, california):
    return simulate(Person(age=age, sex=sex, income=income, occ_class=occ, smoker=smoker, bmi=bmi,
                           chronic=chronic, hobby=hobby, california=california))


# ------------------------------------------------------------------ sidebar: you
with st.sidebar:
    st.header("You")
    age = st.slider("Age", 18, 60, 23)
    sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
    income = st.slider("Annual income now", 30_000, 300_000, 95_000, 5_000, format="$%d")
    occ_label = st.selectbox("Type of work", [
        "Office or professional (data, finance, software)", "Office with some physical work (sales, teaching)",
        "Skilled or light manual (technician, nurse)", "Heavy manual (construction, warehouse)",
        "Medical (doctor, dentist, nurse, pharmacist)"])
    occ = {"Office o": "1", "Office w": "2", "Skilled ": "3", "Heavy ma": "4", "Medical ": "M"}[occ_label[:8]]
    smoker = st.radio("Tobacco", ["No", "Yes, any tobacco"], horizontal=True) != "No"
    bmi = st.selectbox("Body mass index", ["under", "normal", "over", "obese"], index=1,
                       format_func={"under": "Under 18.5", "normal": "18.5 to 24.9", "over": "25 to 29.9",
                                    "obese": "30 or higher"}.get)
    chronic = st.checkbox("Chronic health condition", help="Diabetes, back problems, heart disease, or similar")
    hobby = st.checkbox("High-risk hobbies", help="Climbing, motorsport, skydiving, private flying")
    california = st.checkbox("Live and work in California", value=True,
                             help="Claims run 27% higher on California policies. SDI pays 70–90% of wages "
                                  "for up to 52 weeks, capped at $1,765 a week in 2026.")

    with st.expander("Expected income every 5 years", expanded=True):
        st.caption("Defaults grow about 15% every 5 years until 50, then stay flat. "
                   "Drag each to your own expectations.")
        defaults = default_income_path(age, income)
        if st.button("Reset to defaults"):
            for a in defaults:
                st.session_state.pop(f"inc{a}", None)
        path = {a: st.slider(f"At age {a}", 20_000, 400_000, v, 5_000, key=f"inc{a}", format="$%d")
                for a, v in defaults.items()}

    with st.expander("Assumptions"):
        rate = st.slider("Discount rate (after inflation)", 0.0, 6.0, 3.0, 0.5, format="%.1f%%")
        payout = st.slider("Insurer payout ratio for estimated premiums", 40, 85, 60, 5, format="%d%%",
                           help="Share of premiums the insurer expects to pay back as benefits.")

p = Person(age=age, sex="f" if sex == "Female" else "m", income=income, occ_class=occ, smoker=smoker, bmi=bmi,
           chronic=chronic, hobby=hobby, california=california, income_path=path,
           discount_rate=rate, payout_ratio=payout)

st.title("What a disability could cost you")
tabs = st.tabs(["Policies", "Financial burden", "Worst cases", "Your baseline risk", "Where the numbers come from"])

# ------------------------------------------------------------------ policies (inputs first: they feed every tab)
with tabs[0]:
    st.caption("Set up to three policies to compare. Every tab updates as you change them.")
    policies, cols = [], st.columns(3)
    for i, (col, ep_default) in enumerate(zip(cols, [3, 6, 12])):
        with col:
            st.subheader(NAMES[i + 1])
            ep = st.select_slider("Waiting period", options=list(ELIM_OPTIONS), value=ep_default,
                                  format_func=ELIM_OPTIONS.get, key=f"ep{i}")
            bp = st.select_slider("Pays for up to", options=list(BENEFIT_PERIODS), value=None,
                                  format_func=BENEFIT_PERIODS.get, key=f"bp{i}")
            pct = st.slider("Benefit, % of income", 40, 80, 60, 5, format="%d%%", key=f"pct{i}")
            est = st.checkbox("Use estimated premium", value=True, key=f"est{i}")
            prem = None if est else st.slider("Monthly premium", 0, 1000, 100, 5, format="$%d", key=f"prem{i}")
            offset = st.checkbox("Reduced by SDI", key=f"off{i}")
            fio = st.checkbox("Future increase option", key=f"fio{i}",
                              help="Coverage grows with your income, no new medical exam; premium rises with it.")
            policies.append(Policy(elim_months=ep, benefit_months=bp, benefit_pct=pct, premium=prem,
                                   sdi_offset=offset, future_increase=fio))

sim = cached_sim(p.age, p.sex, p.income, p.occ_class, p.smoker, p.bmi, p.chronic, p.hobby, p.california)
M = Money(p, sim["T"])
base = baseline(sim, p, M)
res = [run_policy(sim, p, pol, base, M) for pol in policies]
losses = [base["loss"]] + [r["loss"] for r in res]

with tabs[0]:
    rows = {
        "Monthly premium": ["—"] + [f"${r['premium']:.0f}" + (" (est.)" if pol.premium is None else "")
                                    + (", then rises" if pol.future_increase else "") for r, pol in zip(res, policies)],
        "Chance it ever pays out": ["—"] + [f"{r['p_claim']:.1%}" for r in res],
        "Expected benefits received": ["—"] + [money(r["exp_benefits"]) for r in res],
        "Expected premiums paid": ["—"] + [money(r["exp_premiums"]) for r in res],
        "Benefits per $1 of premium": ["—"] + [f"${r['exp_benefits']/r['exp_premiums']:.2f}" if r["exp_premiums"] else "—"
                                              for r in res],
        "Total premiums by age 65": ["—"] + [money(r["lifetime_premiums"]) for r in res],
        "Bad case: 1 in 20 loses more than": [money(np.quantile(x, .95)) for x in losses],
        "Severe case: 1 in 100 loses more than": [money(np.quantile(x, .99)) for x in losses],
    }
    st.table(pd.DataFrame(rows, index=NAMES).T)
    st.caption("Expected values are in today's dollars. Loss = income lost − SDI − benefits + premiums, by age 65.")

# ------------------------------------------------------------------ financial burden
with tabs[1]:
    st.caption("Set the yearly income you need to support yourself. If a disability leaves you living on less "
               "than that, this counts it as time spent as a financial burden.")
    c1, c2, c3, c4 = st.columns(4)
    floor = c1.slider("Income you need each year", 20_000, 150_000, 50_000, 5_000, format="$%d")
    min_yr = c2.slider("Being a burden for more than (years)", 0.5, 20.0, 1.0, 0.5, format="%.1f")
    ssdi = c3.slider("SSDI you'd get if approved, per year", 0, 45_000, 18_000, 1_000, format="$%d")
    ssdi_p = c4.slider("Chance Social Security approves", 0, 100, 50, 5, format="%d%%") / 100

    below = [years_below_floor(sim, p, None, M, floor, ssdi, ssdi_p)] + \
            [years_below_floor(sim, p, pol, M, floor, ssdi, ssdi_p) for pol in policies]
    chance = [np.mean(b > min_yr) for b in below]
    avg = [b.mean() for b in below]
    if_hit = [b[b > min_yr].mean() if (b > min_yr).any() else 0 for b in below]
    cost = [0] + [r["exp_premiums"] - r["exp_benefits"] for r in res]
    XY = f"{min_yr:g} year" + ("" if min_yr == 1 else "s")
    pick = min(range(1, 4), key=lambda i: (chance[i], cost[i]))
    pr = res[pick - 1]
    st.markdown(md(f"#### Without a policy, out of 100 people like you, about {per100(chance[0])} would spend more than "
                f"{XY} living below {money(floor)} a year before 65"
                + (f", around {yrs(if_hit[0])} years each" if chance[0] else "") + ". "
                + (f"With {NAMES[pick]}, almost none would." if chance[pick] < 0.005 else
                   f"With {NAMES[pick]}, about {per100(chance[pick])} would.")))
    st.markdown(md(f"{NAMES[pick]} costs ${pr['premium']:.0f} a month, {money(pr['lifetime_premiums'])} in total by age 65. "
                   f"On average, each person pays about {money(cost[pick])} more than they get back."))

    mcols = st.columns(4)
    for i, c in enumerate(mcols):
        if i == 0:
            c.metric(NAMES[0], f"{per100(chance[0])} in 100")
            c.caption(md(f"spend over {XY} below {money(floor)}" + (f", about {yrs(if_hit[0])} years each" if chance[0] else "")))
        else:
            r = res[i - 1]
            c.metric(md(f"{NAMES[i]} · ${r['premium']:.0f}/month") + (" to start" if policies[i-1].future_increase else ""),
                     f"{per100(chance[i])} in 100")
            c.caption(md(f"still spend over {XY} below {money(floor)}. Costs {money(r['lifetime_premiums'])} in total by age 65."))

    horizon = max(2, 65 - age)
    xs = np.linspace(0.5, horizon, 80)
    df = pd.DataFrame([{"Years below your income": x, "Chance of more than this": exceed(b, x), "Option": NAMES[i]}
                       for i, b in enumerate(below) for x in xs])
    chart = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("Years below your income:Q", title="Years living below your income before 65"),
        y=alt.Y("Chance of more than this:Q", axis=alt.Axis(format="%"), title="Chance of more than this"),
        color=alt.Color("Option:N", scale=alt.Scale(domain=NAMES, range=COLORS)),
        strokeDash=alt.condition(alt.datum.Option == "No policy", alt.value([6, 5]), alt.value([1, 0])),
    )
    rule = alt.Chart(pd.DataFrame({"x": [min_yr]})).mark_rule(strokeDash=[2, 4]).encode(x="x:Q")
    st.altair_chart((chart + rule).properties(height=320), width="stretch")

    st.table(pd.DataFrame({
        f"People in 100 below {money(floor)} for over {XY}": [per100(c) for c in chance],
        f"Average years below {money(floor)}, counting everyone": [yrs(a) for a in avg],
        f"Years below {money(floor)}, for those it happens to": [yrs(h) if c else "—" for h, c in zip(if_hit, chance)],
        f"Years below {money(floor)} the policy prevents, on average": ["—"] + [yrs(avg[0] - a) for a in avg[1:]],
        "What you pay beyond what you get back, on average": ["—"] + [money(c) for c in cost[1:]],
        "Monthly premium": ["—"] + [f"${r['premium']:.0f}" for r in res],
        "Total premiums by age 65": ["—"] + [money(r["lifetime_premiums"]) for r in res],
    }, index=NAMES).T)
    st.caption("While disabled, your income is SDI (weeks 2–52, in California), plus SSDI if Social Security approves "
               "you, plus any policy benefits. Averages include the many people who never get disabled.")

# ------------------------------------------------------------------ worst cases
with tabs[2]:
    mark = round(2 * income / 10_000) * 10_000
    p_none = exceed(losses[0], mark)
    best = min(range(1, 4), key=lambda i: exceed(losses[i], mark))
    odds = lambda q: "less than 1 in 50,000" if q <= 0 else f"1 in {1/q:,.0f}" if q < 0.5 else f"{q:.0%}"
    st.markdown(md(f"#### Without a policy, there's a {odds(p_none)} chance a disability costs you more than "
                   f"{money(mark)} before 65. With {NAMES[best]}, it's {odds(exceed(losses[best], mark))}."))
    ratios = [r["exp_benefits"] / r["exp_premiums"] for r in res if r["exp_premiums"]]
    p99 = [np.quantile(x, .99) for x in losses]
    lo, hi = f"${min(ratios):.2f}", f"${max(ratios):.2f}"
    a, b = money(min(p99[1:])), money(max(p99[1:]))
    rng_text = a if a == b else f"{a}–{b}"
    st.markdown(md(f"Each policy is expected to pay back about **{lo if lo == hi else lo + '–' + hi} per $1 of premium**, "
                f"so on average you pay more than you get back. In return, the worst 1-in-100 outcome drops from "
                f"**{money(p99[0])}** to **{rng_text}**. "
                "Whether that trade is worth it is your call."))
    grid = np.logspace(np.log10(5_000), np.log10(max(200_000, np.quantile(losses[0], .9995) * 1.15)), 90)
    df = pd.DataFrame([{"Total loss by 65": x, "Chance of losing more": exceed(l, x), "Option": NAMES[i]}
                       for i, l in enumerate(losses) for x in grid if exceed(l, x) >= 0.001])
    chart = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("Total loss by 65:Q", scale=alt.Scale(type="log", domain=[5_000, float(grid[-1])], nice=False), axis=alt.Axis(format="$,.2~s"),
                title="Total loss by age 65, in today's dollars"),
        y=alt.Y("Chance of losing more:Q", scale=alt.Scale(type="log", domain=[0.001, 1]),
                axis=alt.Axis(format="%"), title="Chance of losing more than this"),
        color=alt.Color("Option:N", scale=alt.Scale(domain=NAMES, range=COLORS)),
        strokeDash=alt.condition(alt.datum.Option == "No policy", alt.value([6, 5]), alt.value([1, 0])),
    )
    st.altair_chart(chart.properties(height=360), width="stretch")
    st.caption("Each line shows the chance your total loss (income lost − SDI − benefits + premiums) exceeds that "
               "amount by 65. Policy lines start high because premiums are a certain cost; a good policy drops far "
               "below the dashed line on the right, where losses are ruinous.")

# ------------------------------------------------------------------ baseline
with tabs[3]:
    st.caption("Before any private policy, from now until 65.")
    b = base
    c = st.columns(3)
    c[0].metric("Any disability over 30 days", f"{b['p_any']:.0%}")
    c[1].metric("Lasting over 3 months", f"{b['p3']:.0%}")
    c[2].metric("Lasting over a year", f"{b['p12']:.1%}")
    c = st.columns(3)
    c[0].metric("Expected income lost", money(b["exp_lost"]))
    c[1].metric("Share of that SDI replaces", f"{b['exp_sdi']/b['exp_lost']:.0%}" if b["exp_lost"] else "—")
    c[2].metric("Worst 1 in 100 without a policy", money(np.quantile(b["loss"], .99)))

# ------------------------------------------------------------------ method
with tabs[4]:
    st.markdown("""
Each simulated life (50,000 of them) moves month by month from your age to 65, facing the chance of dying or of a
disability that lasts over 30 days. Every policy is applied to the same lives.

- **How often disability starts:** Poisson model fitted to the SOA IDEC claim incidence study, 2006–2015
  (25.4 million policy-years, 148,000 claims), by age, sex, occupation class, tobacco, state and policy year.
- **The first three months:** the same model's elimination-period ratios (about 58% of 30-day claims reach 60 days, 38% reach 90).
- **How long it lasts after month 3:** Poisson models of recovery and death fitted to the SOA IDEC claim termination
  study (762,000 claim-years), by time on claim, onset age, sex, occupation class, benefit size and a diagnosis
  drawn from the real claim mix.
- **Deaths while healthy:** SSA Actuarial Note 2026.6. **SDI:** 2026 California rules.
- **Assumptions:** BMI multipliers, chronic condition (1.5×), hobbies (1.15×), the 15% income-growth default,
  and the 60% payout ratio behind estimated premiums.

This is an educational model, not financial advice.
""")
