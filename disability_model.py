"""
Disability insurance policy model (Python version of the dashboard's engine).

Simulates working lives month by month from your age to 65, using models fitted to
Society of Actuaries (SOA) individual disability data, then compares insurance policies.

    from disability_model import Person, Policy, run_model
    results = run_model(Person(age=23, current_income=95000),
                        [Policy(waiting_period_months=3), Policy(waiting_period_months=12)])

Data (CSV files in model_inputs/, written by fit_incidence.py and fit_termination.py):
  INCIDENCE_MODEL      Poisson model of how often claims start (SOA IDEC claim incidence study, 2006-2015)
  RECOVERY_MODEL       Poisson model of recovery while disabled (SOA IDEC claim termination study)
  DEATH_MODEL          Poisson model of death while disabled (SOA IDEC claim termination study)
  DIAGNOSIS_MIX        share of long claims by diagnosis, for men and women
  HEALTHY_DEATH_RATES  healthy-worker death rates by age and sex (SSA Actuarial Note 2026.6)
  2026 federal, Social Security/Medicare and California income taxes (single filer) turn income into take-home pay.
Only numpy is required.
"""
from dataclasses import dataclass, field
from typing import Optional
import csv
import math
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------- model inputs
# The fitted numbers live in plain CSV files in model_inputs/ (open them like any table):
#   incidence_coefficients.csv  how often disabilities start   (written by fit_incidence.py)
#   recovery_coefficients.csv   how fast people recover        (written by fit_termination.py)
#   death_coefficients.csv      how often claimants die        (written by fit_termination.py)
#   diagnosis_mix.csv           share of long claims by diagnosis, for men and women
#   ssa_death_rates.csv         yearly death rates of healthy workers by age (SSA Actuarial Note 2026.6)
# Each coefficient adds to the log of a rate; rate_ratio = exp(coefficient) is the same thing as a multiplier.
MODEL_INPUTS_FOLDER = Path(__file__).resolve().parent / "model_inputs"


def read_model_input_csv(file_name):
    """Read one CSV file from the model_inputs folder.

    Returns a list with one dictionary per row (the header row is not included).
    Each dictionary's keys are the column names from the header row, and its values
    are that row's entries, as text (strings), so numbers must be converted with float().

    Example: read_model_input_csv("recovery_coefficients.csv")[1] ->
        {'variable': 'dur_band', 'level': '4-6m', 'coefficient': '0', 'rate_ratio': '1.0'}
    """
    with open(MODEL_INPUTS_FOLDER / file_name, newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def load_coefficients(file_name):
    """Read a coefficient file into {"intercept": number, variable_name: {level: coefficient}}.

    Example: load_coefficients("incidence_coefficients.csv")["state"]["California"] -> 0.23579
    """
    coefficients = {}
    for row in read_model_input_csv(file_name):
        if row["variable"] == "Intercept":
            coefficients["intercept"] = float(row["coefficient"])
        else:
            coefficients.setdefault(row["variable"], {})[row["level"]] = float(row["coefficient"])
    return coefficients


INCIDENCE_MODEL = load_coefficients("incidence_coefficients.csv")
RECOVERY_MODEL = load_coefficients("recovery_coefficients.csv")
DEATH_MODEL = load_coefficients("death_coefficients.csv")
_diagnosis_rows = read_model_input_csv("diagnosis_mix.csv")
DIAGNOSIS_MIX = {"m": {row["diagnosis"]: float(row["male_share"]) for row in _diagnosis_rows},
                 "f": {row["diagnosis"]: float(row["female_share"]) for row in _diagnosis_rows}}
_ssa_rows = read_model_input_csv("ssa_death_rates.csv")
# Yearly death rate of healthy (not disabled) workers; index 0 is age 20, index 46 is age 66.
HEALTHY_DEATH_RATES = {"m": [float(row["male"]) for row in _ssa_rows],
                       "f": [float(row["female"]) for row in _ssa_rows]}
FIRST_AGE_IN_DEATH_TABLE, LAST_INDEX_IN_DEATH_TABLE = 20, 46

NUMBER_OF_SIMULATED_LIVES = 50_000
RANDOM_SEED = 20260926               # fixed, so every run gives the same results
RETIREMENT_AGE = 65
SDI_MAX_WEEKLY_BENEFIT, SDI_90_PERCENT_WAGE_LIMIT = 1765.0, 1252.30   # 2026 California SDI weekly cap and 90% threshold
SDI_WAITING_PERIOD_MONTHS = 0.23     # SDI's 7-day waiting period, in months
SDI_MAX_MONTHS = 12                  # SDI pays for up to 52 weeks
SSDI_START_MONTH = 5                 # Social Security disability starts after a 5-month wait
SSDI_MIN_DISABILITY_MONTHS = 12      # only disabilities expected to last 12+ months qualify

# Disability risk by BMI, relative to normal weight (18.5-24.9). Pooled risk ratios for all-cause
# disability retirement from Shiri R, Falah-Hassani K, Lallukka T. "Body mass index and the risk of
# disability retirement: a systematic review and meta-analysis." Occup Environ Med. 2020;77(1):48-55.
# https://pubmed.ncbi.nlm.nih.gov/31467042/  (underweight 1.20, 95% CI 1.02-1.41; overweight 1.13,
# 1.07-1.19; obese 1.52, 1.36-1.71). The study measured disability retirement; applying the ratios to
# all new claims is an assumption.
BMI_RISK_MULTIPLIER = {"under": 1.20, "normal": 1.0, "over": 1.13, "obese": 1.52}
# Share of U.S. adults age 20+ in each BMI group, CDC NHANES August 2021-August 2023 (age adjusted):
#   underweight 1.6%  - CDC Health E-Stat 119, https://www.cdc.gov/nchs/data/hestat/hestat119.htm
#   overweight 31.7%, obese 40.3%  - CDC Health E-Stat 111, https://www.cdc.gov/nchs/data/hestat/hestat111.htm
#   normal 26.4%  - not published; the remainder, 100 - 1.6 - 31.7 - 40.3
# Used to average BMI_RISK_MULTIPLIER back to 1 across the population, because the SOA rates already include every BMI.
BMI_POPULATION_SHARE = {"under": 0.016, "normal": 0.264, "over": 0.317, "obese": 0.403}
# Disability risk with a chronic condition: the average of the fully adjusted hazard ratios for disability pension
# for the three conditions the app names (diabetes 1.63, musculoskeletal disorder 1.73, cardiovascular disease 1.35)
# in Ervasti J et al. "Contribution of comorbid conditions to the association between diabetes and disability
# pensions: a population-based nationwide cohort study." Scand J Work Environ Health. 2016;42(3):209-216.
# https://www.sjweh.fi/article/3556  A Swedish population cohort measuring permanent disability pensions; other
# conditions differ (depression was 2.93). Applied in full, because the SOA data comes from underwritten policyholders,
# a group screened to have few chronic conditions.
CHRONIC_CONDITION_DISABILITY_MULTIPLIER = round((1.63 + 1.73 + 1.35) / 3, 2)   # 1.57
# Assumption (no published figure found): high-risk hobbies raise disability risk by 15%.
RISKY_HOBBY_DISABILITY_MULTIPLIER = 1.15

# Death-rate multipliers while healthy. The SSA rates are averages over everyone, so each multiplier is divided by
# its population average: an average person keeps the SSA rate, and only people above or below average move.
# Smoking: all-cause death rate of current vs never smokers, ages 25-79 (men 2.8, 99% CI 2.4-3.1; women 3.0,
# 2.7-3.3), from Jha P et al. "21st-century hazards of smoking and benefits of cessation in the United States."
# N Engl J Med. 2013;368(4):341-350. https://pubmed.ncbi.nlm.nih.gov/23343063/
SMOKER_DEATH_MULTIPLIER = {"m": 2.8, "f": 3.0}
# Share of U.S. adults who currently smoke cigarettes: 9.1% (CDC, National Health Interview Survey, 2025 early release).
# https://www.cdc.gov/nchs/fastats/smoking.htm
SMOKER_POPULATION_SHARE = 0.091
# Body mass index: all-cause death rate relative to BMI 22.5-25, never-smokers without chronic disease, from
# The Global BMI Mortality Collaboration. "Body-mass index and all-cause mortality: individual-participant-data
# meta-analysis of 239 prospective studies in four continents." Lancet. 2016;388(10046):776-786.
# https://pubmed.ncbi.nlm.nih.gov/27423262/  underweight = BMI 15-18.5 (1.51); normal = 1; overweight = average of
# BMI 25-27.5 (1.07) and 27.5-30 (1.20); obese = BMI 30-35 (1.45). Higher BMIs carry more risk (35-40: 1.94,
# 40+: 2.76), so this understates it for them. Averaged over BMI_POPULATION_SHARE.
BMI_DEATH_MULTIPLIER = {"under": 1.51, "normal": 1.0, "over": round((1.07 + 1.20) / 2, 3), "obese": 1.45}
# Assumption (no published figure for a single chronic condition found; for two or more conditions in older adults
# a meta-analysis found 1.73: Nunes BP et al., Arch Gerontol Geriatr. 2016;67:130-138).
CHRONIC_CONDITION_DEATH_MULTIPLIER = 1.4

# Time-on-claim bands used by the recovery and death models: (last claim month in the band, band label)
CLAIM_DURATION_BANDS = [(6, "4-6m"), (12, "7-12m"), (24, "13-24m"), (36, "25-36m"), (60, "37-60m"),
                        (120, "6-10y"), (180, "11-15y"), (240, "16-20y"), (10**9, "21y+")]
WAITING_PERIOD_OPTIONS = {1: "30 days", 2: "60 days", 3: "90 days", 6: "180 days", 12: "365 days", 24: "2 years"}
BENEFIT_PERIOD_OPTIONS = {12: "1 year", 24: "2 years", 60: "5 years", 120: "10 years", None: "To age 65"}
PAYS_UNTIL_65 = 1e9                  # stands in for "no limit" on a benefit period or waiting period

# 2026 taxes, single filer (IRS Rev. Proc. 2025-32; FTB 2026 Schedule X; SSA wage base; EDD SDI rate): https://www.irs.gov/newsroom/irs-releases-tax-inflation-adjustments-for-tax-year-2026-including-amendments-from-the-one-big-beautiful-bill
FEDERAL_STANDARD_DEDUCTION = 16_100
# (top of the bracket, tax rate on income within it)
FEDERAL_TAX_BRACKETS = [(12_400, .10), (50_400, .12), (105_700, .22), (201_775, .24), (256_225, .32),
                        (640_600, .35), (math.inf, .37)]
# https://www.ssa.gov/oact/cola/cbb.html  SOCIAL_SECURITY_TAX_RATE is the share of pay that goes to Social Security.
# SOCIAL_SECURITY_WAGE_BASE is the cap on how much of your pay is subject to Social Security tax.
SOCIAL_SECURITY_TAX_RATE, SOCIAL_SECURITY_WAGE_BASE = 0.062, 184_500

# Below are the Medicare tax and the additional Medicare tax, and the amount of pay
# above which the additional Medicare tax is applied.
MEDICARE_TAX_RATE, ADDITIONAL_MEDICARE_TAX_RATE, ADDITIONAL_MEDICARE_THRESHOLD = 0.0145, 0.009, 200_000

# CALIFORNIA_STANDARD_DEDUCTION is the amount subtracted from your income before the tax brackets are applied.
# CALIFORNIA_PERSONAL_EXEMPTION_CREDIT is the amount taken off the tax.
CALIFORNIA_STANDARD_DEDUCTION, CALIFORNIA_PERSONAL_EXEMPTION_CREDIT = 5_900, 158
CALIFORNIA_TAX_BRACKETS = [(11_456, .01), (27_157, .02), (42_861, .04), (59_498, .06), (75_197, .08),
                           (384_109, .093), (460_927, .103), (768_213, .113), (1_000_000, .123),
                           (math.inf, .133)]     # includes the 1% mental health tax over $1M
CALIFORNIA_SDI_PAYROLL_TAX_RATE = 0.013          # employee SDI payroll tax, no wage cap


# ---------------------------------------------------------------- inputs
@dataclass
class Person:
    age: int = 23
    sex: str = "m"                       # "m" or "f"
    current_income: float = 95_000       # yearly pre-tax income now
    occupation_class: str = "1"          # "1" office, "2" office+physical, "3" light manual, "4" heavy manual, "M" medical
    is_smoker: bool = False
    bmi_group: str = "normal"            # under / normal / over / obese
    has_chronic_condition: bool = False
    has_risky_hobby: bool = False
    lives_in_california: bool = True     # California SDI and California claim rates
    expected_income_by_age: dict = field(default_factory=dict)   # {age: expected income}, e.g. {30: 115000, 40: 155000}
    discount_rate_percent: float = 3.0   # % a year, after inflation
    insurer_payout_percent: float = 60.0  # % of premiums insurers pay back as benefits (sets estimated premiums)
    count_income_after_tax: bool = True  # count lost income as take-home pay (benefits, SDI and SSDI are tax-free)


@dataclass
class Policy:
    waiting_period_months: int = 3       # 1, 2, 3, 6, 12 or 24 months before benefits start
    benefit_period_months: Optional[int] = None   # how long benefits can last; None = until age 65
    benefit_percent_of_income: float = 60.0
    monthly_premium: Optional[float] = None       # None = estimate it from the model
    reduced_by_sdi: bool = False         # benefit reduced by SDI (typical of group policies)
    future_increase_option: bool = False  # coverage (and premium) grows with income


# ---------------------------------------------------------------- helpers
def annual_to_monthly_probability(annual_probability): #checked
    return 1 - (1 - min(annual_probability, 0.99)) ** (1 / 12)


def age_band_label(age): # checked
    """The incidence model's age band for an age, e.g. 23 -> "<30", 41 -> "40-44"."""
    for band_upper_age, band_label in [(30, "<30"), (35, "30-34"), (40, "35-39"), (45, "40-44"), (50, "45-49"),
                                       (55, "50-54"), (60, "55-59"), (65, "60-64")]:
        if age < band_upper_age:
            return band_label
    return "65+"


def policy_year_label(policy_year):
    """The incidence model's policy-year band, e.g. 3 -> "3", 8 -> "6-10"."""
    if policy_year <= 5:
        return str(policy_year)
    return "6-10" if policy_year <= 10 else "11-15" if policy_year <= 15 else "16-20" if policy_year <= 20 else "21+"


def onset_age_group(age_at_onset):
    """The termination models' age-at-onset group."""
    return "<40" if age_at_onset < 40 else "40-49" if age_at_onset < 50 else "50-59" if age_at_onset < 60 else "60+"


def monthly_benefit_band(monthly_benefit):
    """The termination models' benefit-size band, e.g. 4750 -> "2500-4999"."""
    for band_upper_amount, band_label in [(2500, "<2500"), (5000, "2500-4999"), (7500, "5000-7499"),
                                          (10000, "7500-9999"), (15000, "10000-14999"), (20000, "15000-19999")]:
        if monthly_benefit < band_upper_amount:
            return band_label
    return "20000+"


def monthly_sdi_benefit(yearly_income):
    """California SDI paid per month while disabled: 90% or 70% of weekly wages, capped."""
    weekly_wage = yearly_income / 52
    replacement_rate = 0.9 if weekly_wage <= SDI_90_PERCENT_WAGE_LIMIT else 0.7
    return min(SDI_MAX_WEEKLY_BENEFIT, replacement_rate * weekly_wage) * 52 / 12


def tax_from_brackets(taxable_income, brackets):
    tax, bracket_bottom = 0.0, 0.0
    for bracket_top, rate in brackets:
        if taxable_income <= bracket_bottom:
            break
        tax += (min(taxable_income, bracket_top) - bracket_bottom) * rate
        bracket_bottom = bracket_top
    return tax


def yearly_income_taxes(gross_income, lives_in_california=True):
    """2026 taxes on a year's wages for a single filer with the standard deduction."""
    federal_tax = tax_from_brackets(max(0.0, gross_income - FEDERAL_STANDARD_DEDUCTION), FEDERAL_TAX_BRACKETS)
    social_security_and_medicare_tax = (
        SOCIAL_SECURITY_TAX_RATE * min(gross_income, SOCIAL_SECURITY_WAGE_BASE) + MEDICARE_TAX_RATE * gross_income
        + ADDITIONAL_MEDICARE_TAX_RATE * max(0.0, gross_income - ADDITIONAL_MEDICARE_THRESHOLD))
    state_tax = 0.0
    if lives_in_california:
        state_tax = (max(0.0, tax_from_brackets(max(0.0, gross_income - CALIFORNIA_STANDARD_DEDUCTION),
                                                CALIFORNIA_TAX_BRACKETS) - CALIFORNIA_PERSONAL_EXEMPTION_CREDIT)
                     + CALIFORNIA_SDI_PAYROLL_TAX_RATE * gross_income)
    return federal_tax + social_security_and_medicare_tax + state_tax


def yearly_take_home_pay(gross_income, lives_in_california=True):
    """Yearly take-home pay after 2026 federal, Social Security/Medicare and (optionally) California taxes.
    Outside California no state income tax is assumed."""
    return gross_income - yearly_income_taxes(gross_income, lives_in_california)


def default_income_path(age, current_income):
    """About 15% growth every 5 years until 50, then flat (an assumption; override it)."""
    income_by_age, future_age = {}, math.ceil((age + 1) / 5) * 5
    while future_age <= 60:
        years_of_growth = max(0, min(future_age, 50) - age)
        income_by_age[future_age] = round(current_income * 1.15 ** (years_of_growth / 5) / 5000) * 5000
        future_age += 5
    return income_by_age


# ---------------------------------------------------------------- claim-duration tables
def monthly_rate_from_model(model, claim_profile, duration_band):
    """Monthly recovery (or death) rate for one claim, from the fitted coefficients."""
    coefficient = lambda variable, level: model[variable].get(level, 0.0)
    return math.exp(model["intercept"] + coefficient("dur_band", duration_band)
                    + coefficient("onset", claim_profile["onset"]) + coefficient("sex", claim_profile["sex"])
                    + coefficient("occ_class", claim_profile["occ_class"])
                    + coefficient("benefit", claim_profile["benefit"])
                    + coefficient("diagnosis", claim_profile["diagnosis"]))


def monthly_recovery_and_death_chances(person: Person, onset_group, diagnosis, max_claim_months=600):
    """Chance of recovering and of dying in each claim month, for claim months 4..max_claim_months."""
    claim_profile = dict(onset=onset_group, sex="Female" if person.sex == "f" else "Male",
                         occ_class=str(person.occupation_class),
                         benefit=monthly_benefit_band(person.current_income * 0.6 / 12), diagnosis=diagnosis)
    chances_by_band = {
        band_label: (1 - math.exp(-monthly_rate_from_model(RECOVERY_MODEL, claim_profile, band_label)),
                     1 - math.exp(-monthly_rate_from_model(DEATH_MODEL, claim_profile, band_label)))
        for _, band_label in CLAIM_DURATION_BANDS}
    recovery_chance, death_chance = np.zeros(max_claim_months + 1), np.zeros(max_claim_months + 1)
    for claim_month in range(4, max_claim_months + 1):
        band_label = next(label for last_month, label in CLAIM_DURATION_BANDS if claim_month <= last_month)
        recovery_chance[claim_month], death_chance[claim_month] = chances_by_band[band_label]
    return recovery_chance, death_chance


# ---------------------------------------------------------------- simulation
def simulate_lives(person: Person, number_of_lives=NUMBER_OF_SIMULATED_LIVES, seed=RANDOM_SEED):
    """Simulate many lives from the person's age to 65.

    Returns a dict:
      months_to_simulate      months from now until age 65 (at least 12)
      number_of_lives         how many lives were simulated
      episode_life            for each disability: which life it belongs to (0 .. number_of_lives - 1)
      episode_start_month     for each disability: month it started (months from now)
      episode_length_months   for each disability: how long it lasted (cut off at 65)
      death_month             for each life: month of death, or months_to_simulate if alive at 65
    """
    random_generator = np.random.default_rng(seed)
    months_to_simulate = max(12, (RETIREMENT_AGE - person.age) * 12)   # how many months to simulate
    healthy_death_rates = HEALTHY_DEATH_RATES[person.sex]

    # Multipliers on top of the SOA and SSA rates for factors those models don't include.
    # BMI and smoking multipliers are divided by their population averages, so an average person keeps the
    # published rate. The chronic-condition and hobby multipliers are applied in full (see their comments above).
    average_bmi_multiplier = sum(BMI_RISK_MULTIPLIER[group] * BMI_POPULATION_SHARE[group] for group in BMI_POPULATION_SHARE)
    disability_multiplier = (BMI_RISK_MULTIPLIER[person.bmi_group] / average_bmi_multiplier
                             * (CHRONIC_CONDITION_DISABILITY_MULTIPLIER if person.has_chronic_condition else 1)
                             * (RISKY_HOBBY_DISABILITY_MULTIPLIER if person.has_risky_hobby else 1))
    smoker_multiplier = SMOKER_DEATH_MULTIPLIER[person.sex]
    average_smoker_multiplier = 1 - SMOKER_POPULATION_SHARE + SMOKER_POPULATION_SHARE * smoker_multiplier # 1*(share of non-smokers) + multiplier*(share of smokers)
    average_bmi_death_multiplier = sum(BMI_DEATH_MULTIPLIER[group] * BMI_POPULATION_SHARE[group]
                                       for group in BMI_POPULATION_SHARE)
    death_multiplier = ((smoker_multiplier if person.is_smoker else 1) / average_smoker_multiplier
                        * BMI_DEATH_MULTIPLIER[person.bmi_group] / average_bmi_death_multiplier
                        * (CHRONIC_CONDITION_DEATH_MULTIPLIER if person.has_chronic_condition else 1))

    # Log of the yearly claim rate from the parts of the incidence model that don't change with age.
    # Uses the 30-day waiting period level, so it counts every disability lasting over 30 days.
    incidence = INCIDENCE_MODEL
    log_rate_fixed_part = (incidence["intercept"] + incidence["sex"].get("Female" if person.sex == "f" else "Male", 0)
                           + incidence["occ_class"].get(str(person.occupation_class), 0)
                           + incidence["smoker"].get("Yes" if person.is_smoker else "No", 0)
                           + incidence["elim"]["30"]
                           + (incidence["state"]["California"] if person.lives_in_california else 0))

    # Monthly hazards (continuous rates) of starting a disability (>30 days) and of dying while healthy
    disability_hazard, death_hazard = np.zeros(months_to_simulate), np.zeros(months_to_simulate)
    for month in range(months_to_simulate):
        age_now = person.age + month / 12
        yearly_claim_rate = math.exp(log_rate_fixed_part + incidence["age"].get(age_band_label(age_now), 0)
                                     + incidence["policy_year"].get(policy_year_label(month // 12 + 1), 0)
                                     ) * disability_multiplier
        disability_hazard[month] = yearly_claim_rate / 12
        death_table_index = min(max(int(age_now) - FIRST_AGE_IN_DEATH_TABLE, 0), LAST_INDEX_IN_DEATH_TABLE)
        death_hazard[month] = -math.log(1 - annual_to_monthly_probability(
            healthy_death_rates[death_table_index] * death_multiplier))
    cumulative_disability_hazard = np.r_[0, np.cumsum(disability_hazard)]
    cumulative_death_hazard = np.r_[0, np.cumsum(death_hazard)]

    def month_of_next_event(cumulative_hazard, from_month, hazard_to_add):
        """Month when the cumulative hazard, counted from from_month, grows by hazard_to_add (inf if after 65)."""
        target = np.interp(from_month, np.arange(months_to_simulate + 1), cumulative_hazard) + hazard_to_add
        if target >= cumulative_hazard[-1]:
            return np.inf
        month_index = np.searchsorted(cumulative_hazard, target) - 1
        return month_index + (target - cumulative_hazard[month_index]) / (
            cumulative_hazard[month_index + 1] - cumulative_hazard[month_index])

    # Of disabilities lasting over 30 days: share still going at 60 days, and at 90 days given 60.
    share_reaching_60_days = math.exp(incidence["elim"]["60"] - incidence["elim"]["30"])
    share_reaching_90_days_given_60 = math.exp(-incidence["elim"]["60"])
    diagnosis_mix = DIAGNOSIS_MIX[person.sex]
    diagnoses = list(diagnosis_mix)
    diagnosis_weights = np.array(list(diagnosis_mix.values()))
    diagnosis_weights = diagnosis_weights / diagnosis_weights.sum()
    claim_tables = {}               # (onset group, diagnosis) -> (recovery chance, death chance, chance still open)

    episode_life, episode_start_month, episode_length_months = [], [], []
    death_month = np.full(number_of_lives, float(months_to_simulate))
    for life in range(number_of_lives):
        current_month = 0.0
        while current_month < months_to_simulate:
            death_time = month_of_next_event(cumulative_death_hazard, current_month, random_generator.exponential())
            disability_time = month_of_next_event(cumulative_disability_hazard, current_month,
                                                  random_generator.exponential())
            if min(death_time, disability_time) >= months_to_simulate:
                break                                       # healthy until 65
            if death_time < disability_time:
                death_month[life] = death_time
                break
            disability_start, died_while_disabled = disability_time, False
            if random_generator.random() > share_reaching_60_days:
                disability_length = 1 + random_generator.random()      # ends in month 2 ( when share reahcing 60 days is 0.579, ≈ 0.579: about 58% of disabilities that last past 30 days are still going at 60 days. That leaves 42% that end before 60 days. This number comes from the waiting-period coefficients we just went through.
            elif random_generator.random() > share_reaching_90_days_given_60:
                disability_length = 2 + random_generator.random()      # ends in month 3
            else:                                           # lasts past month 3: real claim durations
                onset_group = onset_age_group(person.age + disability_start / 12)
                diagnosis = diagnoses[random_generator.choice(len(diagnoses), p=diagnosis_weights)]
                if (onset_group, diagnosis) not in claim_tables:
                    recovery_chance, death_chance = monthly_recovery_and_death_chances(person, onset_group, diagnosis)
                    chance_still_open = np.cumprod(1 - recovery_chance - death_chance)
                    claim_tables[(onset_group, diagnosis)] = (recovery_chance, death_chance, chance_still_open)
                recovery_chance, death_chance, chance_still_open = claim_tables[(onset_group, diagnosis)]
                random_draw = random_generator.random()
                # first claim month where the chance of still being open drops below the random draw
                end_month = int(np.searchsorted(-chance_still_open[3:], -random_draw)) + 3
                if end_month >= len(chance_still_open):
                    disability_length = 1e9                 # never recovers
                else:
                    disability_length = end_month
                    died_while_disabled = random_generator.random() < death_chance[end_month] / (
                        recovery_chance[end_month] + death_chance[end_month])
            length_before_65 = min(disability_length, months_to_simulate - disability_start)
            episode_life.append(life)
            episode_start_month.append(disability_start)
            episode_length_months.append(length_before_65)
            if died_while_disabled and disability_length <= months_to_simulate - disability_start:
                death_month[life] = disability_start + disability_length
                break
            current_month = math.ceil(disability_start + disability_length)
    return dict(months_to_simulate=months_to_simulate, number_of_lives=number_of_lives,
                episode_life=np.array(episode_life, int), episode_start_month=np.array(episode_start_month),
                episode_length_months=np.array(episode_length_months), death_month=death_month)


# ---------------------------------------------------------------- money
class MoneyOverTime:
    """Income, SDI and discounting for each month from now until 65.

    monthly_gross_income:             pre-tax income each month (policy benefits and SDI are set from this)
    monthly_income_lost_if_disabled:  what a disability takes away each month; take-home pay when
                                      person.count_income_after_tax, since policy benefits (premiums paid with
                                      your own money), SDI and SSDI are tax-free
    monthly_sdi_if_disabled:          California SDI each month (zero outside California)
    The cumulative_discounted_* arrays are running totals in today's dollars, used to add up any span of months.
    """
    def __init__(self, person: Person, months_to_simulate):
        income_by_age = {person.age: person.current_income,
                         **(person.expected_income_by_age or default_income_path(person.age, person.current_income))}
        ages, incomes = zip(*sorted(income_by_age.items()))
        self.months = months_to_simulate
        self.monthly_discount_log_rate = math.log(1 + person.discount_rate_percent / 100) / 12
        age_mid_month = person.age + (np.arange(months_to_simulate) + 0.5) / 12
        yearly_income = np.interp(age_mid_month, ages, incomes)
        self.monthly_gross_income = yearly_income / 12
        self.monthly_income_lost_if_disabled = (
            np.array([yearly_take_home_pay(y, person.lives_in_california) for y in yearly_income]) / 12
            if person.count_income_after_tax else self.monthly_gross_income)
        self.monthly_sdi_if_disabled = (np.array([monthly_sdi_benefit(y) for y in yearly_income])
                                        if person.lives_in_california else np.zeros(months_to_simulate))
        discount_factor = np.exp(-self.monthly_discount_log_rate * (np.arange(months_to_simulate) + 0.5))
        self.cumulative_discounted_income_lost = np.r_[0, np.cumsum(self.monthly_income_lost_if_disabled * discount_factor)]
        self.cumulative_discounted_sdi = np.r_[0, np.cumsum(self.monthly_sdi_if_disabled * discount_factor)]
        # income relative to today's income: used when coverage grows with income (future increase option)
        income_growth_ratio = self.monthly_gross_income / (person.current_income / 12)
        self.cumulative_discounted_income_growth = np.r_[0, np.cumsum(income_growth_ratio * discount_factor)] # running total of "how much has the person's income grown relative to today," month by month, discounted to today's dollars.
        self.total_income_growth_months = float(np.sum(income_growth_ratio))

    def discounted_total_between(self, cumulative, from_month, to_month):
        """Sum of a monthly amount between two months, in today's dollars."""
        from_month, to_month = np.clip(from_month, 0, self.months), np.clip(to_month, 0, self.months)
        month_grid = np.arange(self.months + 1)
        return np.where(to_month > from_month,
                        np.interp(to_month, month_grid, cumulative) - np.interp(from_month, month_grid, cumulative), 0.0)

    def discounted_months_between(self, from_month, to_month): # Months of total episode
        """Value today of $1 a month paid between two months."""
        if self.monthly_discount_log_rate == 0:
            return np.maximum(to_month - from_month, 0)
        rate = self.monthly_discount_log_rate
        return np.where(to_month > from_month, (np.exp(-rate * from_month) - np.exp(-rate * to_month)) / rate, 0.0)

    def value_at_month(self, month, monthly_values):
        return monthly_values[np.clip(np.floor(month).astype(int), 0, self.months - 1)]


def total_per_life(episode_life, values_per_episode, number_of_lives):
    """Add up a value over each life's disabilities (lives with none get 0)."""
    return np.bincount(episode_life, weights=values_per_episode, minlength=number_of_lives)


def no_policy_results(simulation, person, money):
    """Loss per life with no private policy: income lost minus SDI (today's dollars; take-home if count_income_after_tax)."""
    start, length = simulation["episode_start_month"], simulation["episode_length_months"]
    episode_life, number_of_lives = simulation["episode_life"], simulation["number_of_lives"]
    income_lost = money.discounted_total_between(money.cumulative_discounted_income_lost, start, start + length)
    sdi_received = np.where(length > SDI_WAITING_PERIOD_MONTHS,
                            money.discounted_total_between(money.cumulative_discounted_sdi,
                                                           start + SDI_WAITING_PERIOD_MONTHS,
                                                           start + np.minimum(length, SDI_MAX_MONTHS)), 0.0)
    income_lost_per_life = total_per_life(episode_life, income_lost, number_of_lives)
    sdi_per_life = total_per_life(episode_life, sdi_received, number_of_lives)
    longest_disability = np.zeros(number_of_lives)
    np.maximum.at(longest_disability, episode_life, length)
    return dict(loss_per_life=income_lost_per_life - sdi_per_life,
                chance_any_disability=np.mean(np.bincount(episode_life, minlength=number_of_lives) > 0),
                chance_disabled_over_3_months=np.mean(longest_disability > 3),
                chance_disabled_over_12_months=np.mean(longest_disability > 12),
                expected_income_lost=income_lost_per_life.mean(), expected_sdi=sdi_per_life.mean())


def policy_results(simulation, person, policy: Policy, no_policy, money):
    """Loss per life with this policy, its premium, and what it is expected to pay out."""
    start, length = simulation["episode_start_month"], simulation["episode_length_months"]
    episode_life, number_of_lives = simulation["episode_life"], simulation["number_of_lives"]
    waiting = policy.waiting_period_months
    benefit_period = PAYS_UNTIL_65 if policy.benefit_period_months is None else policy.benefit_period_months
    policy_pays = length > waiting                    # disabilities that outlast the waiting period
    monthly_benefit = policy.benefit_percent_of_income / 100 * (
        money.value_at_month(start, money.monthly_gross_income) if policy.future_increase_option
        else person.current_income / 12)
    benefit_start, benefit_end = start + waiting, start + np.minimum(length, waiting + benefit_period)
    benefits = np.where(policy_pays, money.discounted_months_between(benefit_start, benefit_end) * monthly_benefit, 0.0)
    if policy.reduced_by_sdi and person.lives_in_california:
        sdi_overlap_end = np.minimum(benefit_end, start + SDI_MAX_MONTHS) # when benefits or SDI end
        benefits -= np.where(policy_pays & (sdi_overlap_end > benefit_start),
                             money.discounted_months_between(benefit_start, sdi_overlap_end)
                             * np.minimum(monthly_benefit, money.value_at_month(start, money.monthly_sdi_if_disabled)),
                             0.0)
    # Premium months, in today's dollars. With the future increase option the premium grows with income.
    if policy.future_increase_option:
        premium_months = lambda from_month, to_month: money.discounted_total_between(
            money.cumulative_discounted_income_growth, from_month, to_month)
    else:
        premium_months = money.discounted_months_between
    premiums_waived = np.where(policy_pays, premium_months(start + waiting, start + length), 0.0)
    benefits_per_life = total_per_life(episode_life, benefits, number_of_lives)
    premium_months_per_life = np.maximum(0, premium_months(np.zeros(number_of_lives), simulation["death_month"])
                                         - total_per_life(episode_life, premiums_waived, number_of_lives))
    life_ever_paid = np.bincount(episode_life[policy_pays], minlength=number_of_lives) > 0
    if policy.monthly_premium is not None:
        monthly_premium = policy.monthly_premium
    elif premium_months_per_life.mean() > 0:     # set so the policy pays back insurer_payout_percent on average
        monthly_premium = benefits_per_life.mean() / (person.insurer_payout_percent / 100 * premium_months_per_life.mean())
    else:
        monthly_premium = 0.0
    loss_per_life = no_policy["loss_per_life"] - benefits_per_life + monthly_premium * premium_months_per_life
    premium_months_to_65 = money.total_income_growth_months if policy.future_increase_option else money.months
    return dict(loss_per_life=loss_per_life, monthly_premium=monthly_premium, chance_policy_pays=life_ever_paid.mean(),
                expected_benefits=benefits_per_life.mean(),
                expected_premiums=monthly_premium * premium_months_per_life.mean(),
                total_premiums_to_65=monthly_premium * premium_months_to_65)


def years_below_income_floor(simulation, person, policy, money, yearly_income_floor,
                             yearly_ssdi=18_000, ssdi_approval_chance=0.5):
    """Years each life spends with income below yearly_income_floor (SDI + SSDI if approved + benefits).
    All three are tax-free here, so compare them with a take-home floor. policy=None means no policy."""
    start, length = simulation["episode_start_month"], simulation["episode_length_months"]
    episode_life, number_of_lives = simulation["episode_life"], simulation["number_of_lives"]
    # One random number per disability for the SSDI approval decision. A fixed seed gives the same numbers on every
    # call, so no policy and each policy see the same SSDI decisions, and each disability gets its own independent
    # draw. RANDOM_SEED + 1 keeps these separate from the simulation's own random draws.
    ssdi_random_number = np.random.default_rng(RANDOM_SEED + 1).random(len(episode_life))
    gets_ssdi = (ssdi_random_number < ssdi_approval_chance) & (length >= SSDI_MIN_DISABILITY_MONTHS)
    monthly_sdi = money.value_at_month(start, money.monthly_sdi_if_disabled)
    waiting = policy.waiting_period_months if policy else PAYS_UNTIL_65
    benefit_period = ((PAYS_UNTIL_65 if policy.benefit_period_months is None else policy.benefit_period_months)
                      if policy else 0)
    monthly_benefit = (policy.benefit_percent_of_income / 100 * (
        money.value_at_month(start, money.monthly_gross_income) if policy.future_increase_option
        else person.current_income / 12)) if policy else 0 * start
    # Within each disability, income changes only at these months: split it into pieces at them
    change_points = np.stack([np.zeros_like(length), length, np.full_like(length, SDI_WAITING_PERIOD_MONTHS),
                              np.full_like(length, SDI_MAX_MONTHS), np.full_like(length, SSDI_START_MONTH),
                              np.full_like(length, float(waiting)), np.full_like(length, float(waiting + benefit_period))],
                             axis=1)
    change_points = np.sort(np.minimum(change_points, length[:, None]), axis=1)
    months_below_floor = np.zeros_like(length)
    for piece in range(change_points.shape[1] - 1):
        piece_start, piece_end = change_points[:, piece], change_points[:, piece + 1]
        piece_middle = (piece_start + piece_end) / 2
        monthly_income = (np.where((piece_middle >= SDI_WAITING_PERIOD_MONTHS) & (piece_middle < SDI_MAX_MONTHS),
                                   monthly_sdi, 0)
                          + np.where(gets_ssdi & (piece_middle >= SSDI_START_MONTH), yearly_ssdi / 12, 0))
        if policy:
            sdi_reduction = (np.minimum(monthly_benefit, monthly_sdi)
                             if (policy.reduced_by_sdi and person.lives_in_california) else 0)
            policy_payment = monthly_benefit - sdi_reduction * (piece_middle < SDI_MAX_MONTHS)
            monthly_income = monthly_income + np.where((piece_middle >= waiting) & (piece_middle < waiting + benefit_period),
                                                       policy_payment, 0)
        months_below_floor += np.where((piece_end > piece_start) & (monthly_income < yearly_income_floor / 12),
                                       piece_end - piece_start, 0)
    return total_per_life(episode_life, months_below_floor, number_of_lives) / 12


def run_model(person: Person, policies, simulation=None):
    """Everything the dashboard shows, as numbers."""
    simulation = simulation or simulate_lives(person)
    money = MoneyOverTime(person, simulation["months_to_simulate"])
    no_policy = no_policy_results(simulation, person, money)
    return dict(simulation=simulation, money=money, no_policy=no_policy,
                policies=[policy_results(simulation, person, policy, no_policy, money) for policy in policies])


def share_above(values, threshold):
    """Share of lives with a value above threshold."""
    sorted_values = np.sort(values)
    return 1 - np.searchsorted(sorted_values, threshold, side="right") / len(sorted_values)


if __name__ == "__main__":
    import time
    start_time = time.time()
    person = Person()
    policies = [Policy(waiting_period_months=3), Policy(waiting_period_months=6), Policy(waiting_period_months=12)]
    results = run_model(person, policies)
    no_policy = results["no_policy"]
    print(f"simulated in {time.time() - start_time:.1f}s; disability >30d {no_policy['chance_any_disability']:.0%}, "
          f">1y {no_policy['chance_disabled_over_12_months']:.1%}")
    for name, result in zip("ABC", results["policies"]):
        print(name, f"premium ${result['monthly_premium']:.0f}/mo  pays out {result['chance_policy_pays']:.1%}  "
              f"1-in-100 loss ${np.quantile(result['loss_per_life'], .99):,.0f}")
    print("no policy 1-in-100 loss", f"${np.quantile(no_policy['loss_per_life'], .99):,.0f}")
    print(f"take-home on $95k in California: ${yearly_take_home_pay(95_000):,.0f}")
