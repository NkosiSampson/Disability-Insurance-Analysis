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

from disability_model import (BENEFIT_PERIOD_OPTIONS, WAITING_PERIOD_OPTIONS, MoneyOverTime, Person, Policy,
                              default_income_path, no_policy_results, policy_results, share_above, simulate_lives,
                              years_below_income_floor)

st.set_page_config(page_title="Disability policy explorer", page_icon="🛡️", layout="wide")

OPTION_NAMES = ["No policy", "Policy A", "Policy B", "Policy C"]
OPTION_COLORS = ["#B3362B", "#16707A", "#6A4DB0", "#A17010"]


def format_money(amount):
    """$1.2M, $45k, $300, with a minus sign for negatives."""
    size = abs(amount)
    text = f"{size/1e6:.1f}M" if size >= 1e6 else f"{size/1e3:.0f}k" if size >= 1e3 else f"{size:.0f}"
    return ("−" if amount < 0 else "") + "$" + text


def people_in_100(share):
    """A share as 'people out of 100', e.g. 0.075 -> '7.5'."""
    per_hundred = share * 100
    if per_hundred <= 0:
        return "0"
    if per_hundred < 1:
        return "fewer than 1"
    return f"{round(per_hundred * 2) / 2:g}" if per_hundred < 10 else f"{per_hundred:.0f}"


def escape_dollars(text):
    """Escape dollar signs so Streamlit doesn't read them as math."""
    return text.replace("\\$", "$").replace("$", "\\$")


def format_years(years):
    return f"{years:.0f}" if years >= 10 else f"{years:.1f}" if years >= 1 else f"{years:.2f}"


@st.cache_data(show_spinner="Simulating 50,000 working lives…")
def cached_simulation(age, sex, current_income, occupation_class, is_smoker, bmi_group, has_chronic_condition,
                      has_risky_hobby, lives_in_california):
    return simulate_lives(Person(age=age, sex=sex, current_income=current_income, occupation_class=occupation_class,
                                 is_smoker=is_smoker, bmi_group=bmi_group, has_chronic_condition=has_chronic_condition,
                                 has_risky_hobby=has_risky_hobby, lives_in_california=lives_in_california))


# ------------------------------------------------------------------ sidebar: you
with st.sidebar:
    st.header("You")
    age = st.slider("Age", 18, 60, 23)
    sex_choice = st.radio("Sex", ["Male", "Female"], horizontal=True)
    current_income = st.slider("Annual income now", 30_000, 300_000, 95_000, 5_000, format="$%d")
    occupation_label = st.selectbox("Type of work", [
        "Office or professional (data, finance, software)", "Office with some physical work (sales, teaching)",
        "Skilled or light manual (technician, nurse)", "Heavy manual (construction, warehouse)",
        "Medical (doctor, dentist, nurse, pharmacist)"])
    occupation_class = {"Office o": "1", "Office w": "2", "Skilled ": "3", "Heavy ma": "4",
                        "Medical ": "M"}[occupation_label[:8]]
    is_smoker = st.radio("Tobacco", ["No", "Yes, any tobacco"], horizontal=True) != "No"
    bmi_group = st.selectbox("Body mass index", ["under", "normal", "over", "obese"], index=1,
                             format_func={"under": "Under 18.5", "normal": "18.5 to 24.9", "over": "25 to 29.9",
                                          "obese": "30 or higher"}.get)
    has_chronic_condition = st.checkbox("Chronic health condition",
                                        help="Diabetes, back problems, heart disease, or similar")
    has_risky_hobby = st.checkbox("High-risk hobbies", help="Climbing, motorsport, skydiving, private flying")
    lives_in_california = st.checkbox("Live and work in California", value=True,
                                      help="Claims run 27% higher on California policies. SDI pays 70–90% of wages "
                                           "for up to 52 weeks, capped at $1,765 a week in 2026.")

    with st.expander("Expected income every 5 years", expanded=True):
        st.caption("Defaults grow about 15% every 5 years until 50, then stay flat. "
                   "Drag each to your own expectations.")
        default_incomes = default_income_path(age, current_income)
        if st.button("Reset to defaults"):
            for future_age in default_incomes:
                st.session_state.pop(f"income_at_{future_age}", None)
        expected_income_by_age = {
            future_age: st.slider(f"At age {future_age}", 20_000, 400_000, default_income, 5_000,
                                  key=f"income_at_{future_age}", format="$%d")
            for future_age, default_income in default_incomes.items()}

    with st.expander("Assumptions"):
        discount_rate_percent = st.slider(
            "Discount rate (after inflation)", 0.0, 6.0, 3.0, 0.5, format="%.1f%%",
            help="Turns money paid or lost in future years into today's dollars, because money you "
                 "have now could be invested and grow. At 3%, losing \\$1,000 thirty years from now "
                 "counts as about \\$410 today. It's after inflation because incomes here are in "
                 "today's dollars too. A higher rate makes losses far in the future, like a long "
                 "disability at a young age, look smaller, so insurance looks less valuable. At 0%, a "
                 "future dollar counts the same as a dollar today. The \"Total premiums by age 65\" row "
                 "is not discounted.")
        insurer_payout_percent = st.slider("Insurer payout ratio for estimated premiums", 40, 85, 60, 5, format="%d%%",
                                           help="Share of premiums the insurer expects to pay back as benefits.")
        count_income_after_tax = st.checkbox(
            "Count income after taxes (take-home pay)", value=True,
            help="A disability takes away your take-home pay, not your pre-tax salary. "
                 "Policy benefits (when you pay the premiums yourself), SDI and SSDI are "
                 "tax-free, so this compares like with like. Uses 2026 federal, Social "
                 "Security/Medicare and California taxes for a single filer. Uncheck to "
                 "count everything before taxes.")

person = Person(age=age, sex="f" if sex_choice == "Female" else "m", current_income=current_income,
                occupation_class=occupation_class, is_smoker=is_smoker, bmi_group=bmi_group,
                has_chronic_condition=has_chronic_condition, has_risky_hobby=has_risky_hobby,
                lives_in_california=lives_in_california, expected_income_by_age=expected_income_by_age,
                discount_rate_percent=discount_rate_percent, insurer_payout_percent=insurer_payout_percent,
                count_income_after_tax=count_income_after_tax)

st.title("What a disability could cost you")
INCOME_BASIS = "take-home" if count_income_after_tax else "pre-tax"
tabs = st.tabs(["Policies", "Financial burden", "Worst cases", "Your baseline risk", "Where the numbers come from"])
policies_tab, burden_tab, worst_cases_tab, baseline_tab, method_tab = tabs

# ------------------------------------------------------------------ policies (inputs first: they feed every tab)
with policies_tab:
    st.caption("Set up to three policies to compare. Every tab updates as you change them.")
    policies, policy_columns = [], st.columns(3)
    for policy_number, (column, default_waiting_period) in enumerate(zip(policy_columns, [3, 6, 12])):
        with column:
            st.subheader(OPTION_NAMES[policy_number + 1])
            waiting_period_months = st.select_slider(
                "Waiting period", options=list(WAITING_PERIOD_OPTIONS), value=default_waiting_period,
                format_func=WAITING_PERIOD_OPTIONS.get, key=f"waiting_period_{policy_number}")
            benefit_period_months = st.select_slider(
                "Pays for up to", options=list(BENEFIT_PERIOD_OPTIONS), value=None,
                format_func=BENEFIT_PERIOD_OPTIONS.get, key=f"benefit_period_{policy_number}")
            benefit_percent = st.slider("Benefit, % of income", 40, 80, 60, 5, format="%d%%",
                                        key=f"benefit_percent_{policy_number}")
            use_estimated_premium = st.checkbox("Use estimated premium", value=True,
                                                key=f"use_estimated_premium_{policy_number}")
            monthly_premium = None if use_estimated_premium else st.slider(
                "Monthly premium", 0, 1000, 100, 5, format="$%d", key=f"monthly_premium_{policy_number}")
            reduced_by_sdi = st.checkbox("Reduced by SDI", key=f"reduced_by_sdi_{policy_number}")
            future_increase_option = st.checkbox(
                "Future increase option", key=f"future_increase_{policy_number}",
                help="Coverage grows with your income, no new medical exam; premium rises with it.")
            policies.append(Policy(waiting_period_months=waiting_period_months,
                                   benefit_period_months=benefit_period_months,
                                   benefit_percent_of_income=benefit_percent, monthly_premium=monthly_premium,
                                   reduced_by_sdi=reduced_by_sdi, future_increase_option=future_increase_option))

simulation = cached_simulation(person.age, person.sex, person.current_income, person.occupation_class,
                               person.is_smoker, person.bmi_group, person.has_chronic_condition,
                               person.has_risky_hobby, person.lives_in_california)
money = MoneyOverTime(person, simulation["months_to_simulate"])
no_policy = no_policy_results(simulation, person, money)
policy_outcomes = [policy_results(simulation, person, policy, no_policy, money) for policy in policies]
loss_per_life_by_option = [no_policy["loss_per_life"]] + [outcome["loss_per_life"] for outcome in policy_outcomes]

with policies_tab:
    table_rows = {
        "Monthly premium": ["—"] + [f"${outcome['monthly_premium']:.0f}"
                                    + (" (est.)" if policy.monthly_premium is None else "")
                                    + (", then rises" if policy.future_increase_option else "")
                                    for outcome, policy in zip(policy_outcomes, policies)],
        "Chance it ever pays out": ["—"] + [f"{outcome['chance_policy_pays']:.1%}" for outcome in policy_outcomes],
        "Expected benefits received": ["—"] + [format_money(outcome["expected_benefits"]) for outcome in policy_outcomes],
        "Expected premiums paid": ["—"] + [format_money(outcome["expected_premiums"]) for outcome in policy_outcomes],
        "Benefits per $1 of premium": ["—"] + [f"${outcome['expected_benefits']/outcome['expected_premiums']:.2f}"
                                              if outcome["expected_premiums"] else "—" for outcome in policy_outcomes],
        "Total premiums by age 65": ["—"] + [format_money(outcome["total_premiums_to_65"]) for outcome in policy_outcomes],
        "Bad case: 1 in 20 loses more than": [format_money(np.quantile(losses, .95))
                                              for losses in loss_per_life_by_option],
        "Severe case: 1 in 100 loses more than": [format_money(np.quantile(losses, .99))
                                                  for losses in loss_per_life_by_option],
        "Very severe case: 1 in 200 loses more than": [format_money(np.quantile(losses, .995))
                                                       for losses in loss_per_life_by_option],
        "Extreme case: 1 in 2,000 loses more than": [format_money(np.quantile(losses, .9995))
                                                     for losses in loss_per_life_by_option],
    }
    st.table(pd.DataFrame(table_rows, index=OPTION_NAMES).T)
    st.caption(("Income lost is take-home pay (after 2026 taxes); benefits and SDI are tax-free. "
                if count_income_after_tax else "Everything is counted before taxes. ")
               + "Expected values are in today's dollars. Loss = income lost − SDI − benefits + premiums, by age 65.")

# ------------------------------------------------------------------ financial burden
with burden_tab:
    st.caption("Set the yearly income you need to support yourself. If a disability leaves you living on less "
               "than that, this counts it as time spent as a financial burden.")
    floor_column, years_column, ssdi_column, approval_column = st.columns(4)
    yearly_income_floor = floor_column.slider(
        "Take-home income you need each year" if count_income_after_tax else "Income you need each year",
        20_000, 150_000, 50_000, 5_000, format="$%d")
    burden_years_threshold = years_column.slider("Being a burden for more than (years)", 0.5, 20.0, 1.0, 0.5,
                                                 format="%.1f")
    yearly_ssdi = ssdi_column.slider("SSDI you'd get if approved, per year", 0, 45_000, 18_000, 1_000, format="$%d")
    ssdi_approval_chance = approval_column.slider("Chance Social Security approves", 0, 100, 50, 5,
                                                  format="%d%%") / 100

    years_below_floor_by_option = [
        years_below_income_floor(simulation, person, policy, money, yearly_income_floor, yearly_ssdi,
                                 ssdi_approval_chance)
        for policy in [None] + policies]
    chance_of_burden = [np.mean(years > burden_years_threshold) for years in years_below_floor_by_option]
    average_years_below = [years.mean() for years in years_below_floor_by_option]
    years_below_if_burdened = [years[years > burden_years_threshold].mean()
                               if (years > burden_years_threshold).any() else 0
                               for years in years_below_floor_by_option]
    net_cost_of_policy = [0] + [outcome["expected_premiums"] - outcome["expected_benefits"]
                                for outcome in policy_outcomes]
    threshold_text = f"{burden_years_threshold:g} year" + ("" if burden_years_threshold == 1 else "s")
    best_option = min(range(1, 4), key=lambda option: (chance_of_burden[option], net_cost_of_policy[option]))
    best_outcome = policy_outcomes[best_option - 1]
    st.markdown(escape_dollars(
        f"#### Without a policy, out of 100 people like you, about {people_in_100(chance_of_burden[0])} would spend "
        f"more than {threshold_text} living below {format_money(yearly_income_floor)} a year before 65"
        + (f", around {format_years(years_below_if_burdened[0])} years each" if chance_of_burden[0] else "") + ". "
        + (f"With {OPTION_NAMES[best_option]}, almost none would." if chance_of_burden[best_option] < 0.005 else
           f"With {OPTION_NAMES[best_option]}, about {people_in_100(chance_of_burden[best_option])} would.")))
    st.markdown(escape_dollars(
        f"{OPTION_NAMES[best_option]} costs ${best_outcome['monthly_premium']:.0f} a month, "
        f"{format_money(best_outcome['total_premiums_to_65'])} in total by age 65. "
        f"On average, each person pays about {format_money(net_cost_of_policy[best_option])} more than they get back."))

    metric_columns = st.columns(4)
    for option, column in enumerate(metric_columns):
        if option == 0:
            column.metric(OPTION_NAMES[0], f"{people_in_100(chance_of_burden[0])} in 100")
            column.caption(escape_dollars(
                f"spend over {threshold_text} below {format_money(yearly_income_floor)}"
                + (f", about {format_years(years_below_if_burdened[0])} years each" if chance_of_burden[0] else "")))
        else:
            outcome = policy_outcomes[option - 1]
            column.metric(escape_dollars(f"{OPTION_NAMES[option]} · ${outcome['monthly_premium']:.0f}/month")
                          + (" to start" if policies[option - 1].future_increase_option else ""),
                          f"{people_in_100(chance_of_burden[option])} in 100")
            column.caption(escape_dollars(
                f"still spend over {threshold_text} below {format_money(yearly_income_floor)}. "
                f"Costs {format_money(outcome['total_premiums_to_65'])} in total by age 65."))

    years_to_65 = max(2, 65 - age)
    years_axis = np.linspace(0.5, years_to_65, 80)
    chart_data = pd.DataFrame([{"Years below your income": years, "Chance of more than this": share_above(option_years, years),
                                "Option": OPTION_NAMES[option]}
                               for option, option_years in enumerate(years_below_floor_by_option)
                               for years in years_axis])
    chart = alt.Chart(chart_data).mark_line(strokeWidth=2.5).encode(
        x=alt.X("Years below your income:Q", title="Years living below your income before 65"),
        y=alt.Y("Chance of more than this:Q", axis=alt.Axis(format="%"), title="Chance of more than this"),
        color=alt.Color("Option:N", scale=alt.Scale(domain=OPTION_NAMES, range=OPTION_COLORS)),
        strokeDash=alt.condition(alt.datum.Option == "No policy", alt.value([6, 5]), alt.value([1, 0])),
    )
    threshold_line = alt.Chart(pd.DataFrame({"x": [burden_years_threshold]})).mark_rule(strokeDash=[2, 4]).encode(x="x:Q")
    st.altair_chart((chart + threshold_line).properties(height=320), width="stretch")

    floor_text = format_money(yearly_income_floor)
    st.table(pd.DataFrame({
        f"People in 100 below {floor_text} for over {threshold_text}": [people_in_100(c) for c in chance_of_burden],
        f"Average years below {floor_text}, counting everyone": [format_years(a) for a in average_years_below],
        f"Years below {floor_text}, for those it happens to": [format_years(years) if chance else "—"
                                                               for years, chance in zip(years_below_if_burdened,
                                                                                        chance_of_burden)],
        f"Years below {floor_text} the policy prevents, on average": ["—"] + [
            format_years(average_years_below[0] - average) for average in average_years_below[1:]],
        "What you pay beyond what you get back, on average": ["—"] + [format_money(cost)
                                                                       for cost in net_cost_of_policy[1:]],
        "Monthly premium": ["—"] + [f"${outcome['monthly_premium']:.0f}" for outcome in policy_outcomes],
        "Total premiums by age 65": ["—"] + [format_money(outcome["total_premiums_to_65"]) for outcome in policy_outcomes],
    }, index=OPTION_NAMES).T)
    st.caption("While disabled, your income is SDI (weeks 2–52, in California), plus SSDI if Social Security approves "
               "you, plus any policy benefits, all tax-free. Averages include the many people who never get disabled.")

# ------------------------------------------------------------------ worst cases
with worst_cases_tab:
    loss_mark = round(2 * current_income / 10_000) * 10_000      # about two years of income
    chance_over_mark_no_policy = share_above(loss_per_life_by_option[0], loss_mark)
    best_option = min(range(1, 4), key=lambda option: share_above(loss_per_life_by_option[option], loss_mark))

    def odds_text(chance):
        return "less than 1 in 50,000" if chance <= 0 else f"1 in {1/chance:,.0f}" if chance < 0.5 else f"{chance:.0%}"

    st.markdown(escape_dollars(
        f"#### Without a policy, there's a {odds_text(chance_over_mark_no_policy)} chance a disability costs you more than "
        f"{format_money(loss_mark)} before 65. With {OPTION_NAMES[best_option]}, it's "
        f"{odds_text(share_above(loss_per_life_by_option[best_option], loss_mark))}."))
    payback_per_dollar = [outcome["expected_benefits"] / outcome["expected_premiums"]
                          for outcome in policy_outcomes if outcome["expected_premiums"]]
    one_in_100_loss = [np.quantile(losses, .99) for losses in loss_per_life_by_option]
    lowest_payback, highest_payback = f"${min(payback_per_dollar):.2f}", f"${max(payback_per_dollar):.2f}"
    lowest_policy_loss, highest_policy_loss = format_money(min(one_in_100_loss[1:])), format_money(max(one_in_100_loss[1:]))
    policy_loss_range = (lowest_policy_loss if lowest_policy_loss == highest_policy_loss
                         else f"{lowest_policy_loss}–{highest_policy_loss}")
    payback_range = (lowest_payback if lowest_payback == highest_payback
                     else lowest_payback + "–" + highest_payback)
    st.markdown(escape_dollars(
        f"Each policy is expected to pay back about **{payback_range} per $1 of premium**, "
        f"so on average you pay more than you get back. In return, the worst 1-in-100 outcome drops from "
        f"**{format_money(one_in_100_loss[0])}** to **{policy_loss_range}**. "
        "Whether that trade is worth it is your call."))
    loss_axis = np.logspace(np.log10(5_000),
                            np.log10(max(200_000, np.quantile(loss_per_life_by_option[0], .9995) * 1.15)), 90)
    chart_data = pd.DataFrame([{"Total loss by 65": loss, "Chance of losing more": share_above(losses, loss),
                                "Option": OPTION_NAMES[option]}
                               for option, losses in enumerate(loss_per_life_by_option)
                               for loss in loss_axis if share_above(losses, loss) >= 0.001])
    chart = alt.Chart(chart_data).mark_line(strokeWidth=2.5).encode(
        x=alt.X("Total loss by 65:Q", scale=alt.Scale(type="log", domain=[5_000, float(loss_axis[-1])], nice=False),
                axis=alt.Axis(format="$,.2~s"), title=f"Total loss by age 65, in today's dollars ({INCOME_BASIS})"),
        y=alt.Y("Chance of losing more:Q", scale=alt.Scale(type="log", domain=[0.001, 1]),
                axis=alt.Axis(format="%"), title="Chance of losing more than this"),
        color=alt.Color("Option:N", scale=alt.Scale(domain=OPTION_NAMES, range=OPTION_COLORS)),
        strokeDash=alt.condition(alt.datum.Option == "No policy", alt.value([6, 5]), alt.value([1, 0])),
    )
    st.altair_chart(chart.properties(height=360), width="stretch")
    st.caption("Each line shows the chance your total loss (income lost − SDI − benefits + premiums) exceeds that "
               "amount by 65. Policy lines start high because premiums are a certain cost; a good policy drops far "
               "below the dashed line on the right, where losses are ruinous.")

# ------------------------------------------------------------------ baseline
with baseline_tab:
    st.caption("Before any private policy, from now until 65.")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Any disability over 30 days", f"{no_policy['chance_any_disability']:.0%}")
    metric_columns[1].metric("Lasting over 3 months", f"{no_policy['chance_disabled_over_3_months']:.0%}")
    metric_columns[2].metric("Lasting over a year", f"{no_policy['chance_disabled_over_12_months']:.1%}")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Expected take-home pay lost" if count_income_after_tax else "Expected income lost",
                             format_money(no_policy["expected_income_lost"]))
    metric_columns[1].metric("Share of that SDI replaces",
                             f"{no_policy['expected_sdi']/no_policy['expected_income_lost']:.0%}"
                             if no_policy["expected_income_lost"] else "—")
    metric_columns[2].metric("Worst 1 in 100 without a policy", format_money(np.quantile(no_policy["loss_per_life"], .99)))

# ------------------------------------------------------------------ method
with method_tab:
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
- **Taxes:** with "Count income after taxes" on (the default), income lost is take-home pay after 2026 federal
  income tax, Social Security and Medicare, and California income tax and SDI payroll tax, for a single filer with
  the standard deduction. Policy benefits are a percentage of pre-tax income but are tax-free when you pay the
  premiums yourself, as are SDI and (at these amounts) SSDI. Employer-paid policies would be taxable.
- **BMI:** risk multipliers from a meta-analysis of disability retirement (Shiri et al., 2020), averaged over the
  U.S. adult BMI mix from CDC NHANES 2021–2023.
- **Assumptions:** chronic condition (1.5×), hobbies (1.15×), the 15% income-growth default,
  and the 60% payout ratio behind estimated premiums.

This is an educational model, not financial advice.
""")
