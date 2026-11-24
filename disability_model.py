"""
Disability insurance policy model (Python version of the dashboard's engine).

Simulates working lives month by month from your age to 65, using models fitted to
Society of Actuaries (SOA) individual disability data, then compares insurance policies.

    from disability_model import Person, Policy, run_model
    res = run_model(Person(age=23, income=95000), [Policy(elim_months=3), Policy(elim_months=12)])

Data (CSV files in model_inputs/, written by fit_incidence.py and fit_termination.py):
  INC  Poisson model of how often claims start (SOA IDEC claim incidence study, 2006-2015)
  GLM  Poisson models of recovery and death while disabled (SOA IDEC claim termination study)
  SSA  healthy-worker death rates by age and sex (SSA Actuarial Note 2026.6)
  TAX  2026 federal, Social Security/Medicare and California income taxes (single filer),
       used to turn income into take-home pay
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
INPUTS = Path(__file__).resolve().parent / "model_inputs"

# short names used inside the model for each variable in the CSV files
INC_KEYS = {"age": "age", "sex": "sex", "occ_class": "cls", "smoker": "smoker", "elim": "ep",
            "benefit_period": "bp", "state": "state", "policy_year": "pdur", "underwriting": "uw", "market": "mkt"}
GLM_KEYS = {"dur_band": "dur", "onset": "onset", "sex": "sex", "occ_class": "cls", "benefit": "ben",
            "diagnosis": "diag", "elim": "elim"}


def _read_csv(name):
    """Read one CSV file from the model_inputs folder.

    Returns a list with one dictionary per row (the header row is not included).
    Each dictionary's keys are the column names from the header row, and its values
    are that row's entries, as text (strings), so numbers must be converted with float().

    Example: _read_csv("recovery_coefficients.csv")[1] ->
        {'variable': 'dur_band', 'level': '4-6m', 'coefficient': '0', 'rate_ratio': '1.0'}
    """
    with open(INPUTS / name, newline="") as f:
        return list(csv.DictReader(f))


def load_coefficients(name, keys):
    """Read a coefficient file into {"b0": intercept, short_name: {level: coefficient}}."""
    model = {short: {} for short in keys.values()}
    for row in _read_csv(name):
        if row["variable"] == "Intercept":
            model["b0"] = float(row["coefficient"])
        elif row["variable"] in keys:
            model[keys[row["variable"]]][row["level"]] = float(row["coefficient"])
    return model


INC = load_coefficients("incidence_coefficients.csv", INC_KEYS)
GLM = {"rec": load_coefficients("recovery_coefficients.csv", GLM_KEYS),
       "dth": load_coefficients("death_coefficients.csv", GLM_KEYS),
       "mix": {"m": {r["diagnosis"]: float(r["male_share"]) for r in _read_csv("diagnosis_mix.csv")},
               "f": {r["diagnosis"]: float(r["female_share"]) for r in _read_csv("diagnosis_mix.csv")}}}
_ssa = _read_csv("ssa_death_rates.csv")
SSA = {"m": {"qa": [float(r["male"]) for r in _ssa]},      # qa[0] is age 20, qa[46] is age 66
       "f": {"qa": [float(r["female"]) for r in _ssa]}}

N_LIVES = 50_000
SEED = 20260926
SDI_CAP, SDI_90_LIMIT = 1765.0, 1252.30            # 2026 California SDI weekly cap and 90% threshold
# Disability risk by BMI, relative to normal weight (18.5-24.9). Pooled risk ratios for all-cause
# disability retirement from Shiri R, Falah-Hassani K, Lallukka T. "Body mass index and the risk of
# disability retirement: a systematic review and meta-analysis." Occup Environ Med. 2020;77(1):48-55.
# https://pubmed.ncbi.nlm.nih.gov/31467042/  (underweight 1.20, 95% CI 1.02-1.41; overweight 1.13,
# 1.07-1.19; obese 1.52, 1.36-1.71). The study measured disability retirement; applying the ratios to
# all new claims is an assumption.
BMI_RR = {"under": 1.20, "normal": 1.0, "over": 1.13, "obese": 1.52}
# Share of U.S. adults age 20+ in each BMI group, CDC NHANES August 2021-August 2023 (age adjusted):
#   underweight 1.6%  - CDC Health E-Stat 119, https://www.cdc.gov/nchs/data/hestat/hestat119.htm
#   overweight 31.7%, obese 40.3%  - CDC Health E-Stat 111, https://www.cdc.gov/nchs/data/hestat/hestat111.htm
#   normal 26.4%  - not published; the remainder, 100 - 1.6 - 31.7 - 40.3
# Used to average BMI_RR back to 1 across the population, because the SOA rates already include every BMI.
BMI_MIX = {"under": 0.016, "normal": 0.264, "over": 0.317, "obese": 0.403}
LT_BANDS = [(6, "4-6m"), (12, "7-12m"), (24, "13-24m"), (36, "25-36m"), (60, "37-60m"),
            (120, "6-10y"), (180, "11-15y"), (240, "16-20y"), (10**9, "21y+")]
ELIM_OPTIONS = {1: "30 days", 2: "60 days", 3: "90 days", 6: "180 days", 12: "365 days", 24: "2 years"}
BENEFIT_PERIODS = {12: "1 year", 24: "2 years", 60: "5 years", 120: "10 years", None: "To age 65"}

# 2026 taxes, single filer (IRS Rev. Proc. 2025-32; FTB 2026 Schedule X; SSA wage base; EDD SDI rate)
FED_STD = 16_100
FED_BRACKETS = [(12_400, .10), (50_400, .12), (105_700, .22), (201_775, .24), (256_225, .32), (640_600, .35),
                (math.inf, .37)]
SS_RATE, SS_WAGE_BASE = 0.062, 184_500
MEDICARE_RATE, MEDICARE_EXTRA, MEDICARE_EXTRA_OVER = 0.0145, 0.009, 200_000
CA_STD, CA_CREDIT = 5_900, 158
CA_BRACKETS = [(11_456, .01), (27_157, .02), (42_861, .04), (59_498, .06), (75_197, .08), (384_109, .093),
               (460_927, .103), (768_213, .113), (1_000_000, .123), (math.inf, .133)]   # incl. 1% over $1M
CA_SDI_RATE = 0.013                                 # employee SDI payroll tax, no wage cap


# ---------------------------------------------------------------- inputs
@dataclass
class Person:
    age: int = 23
    sex: str = "m"                 # "m" or "f"
    income: float = 95_000         # income now
    occ_class: str = "1"           # "1" office, "2" office+physical, "3" light manual, "4" heavy manual, "M" medical
    smoker: bool = False
    bmi: str = "normal"            # under / normal / over / obese
    chronic: bool = False
    hobby: bool = False
    california: bool = True        # California SDI and CA claim rates
    income_path: dict = field(default_factory=dict)   # {age: expected income}, e.g. {30: 115000, 40: 155000}
    discount_rate: float = 3.0     # % a year, after inflation
    payout_ratio: float = 60.0     # % of premiums insurers pay back (for estimated premiums)
    after_tax: bool = True         # count lost income as take-home pay (benefits, SDI and SSDI are tax-free)


@dataclass
class Policy:
    elim_months: int = 3           # waiting period: 1, 2, 3, 6, 12 or 24 months
    benefit_months: Optional[int] = None   # None = to age 65
    benefit_pct: float = 60.0      # % of income
    premium: Optional[float] = None   # monthly premium; None = estimate from the model
    sdi_offset: bool = False       # benefit reduced by SDI (typical of group policies)
    future_increase: bool = False  # coverage (and premium) grows with income


# ---------------------------------------------------------------- helpers
def monthly_prob(q):                     # annual probability -> monthly probability
    return 1 - (1 - min(q, 0.99)) ** (1 / 12)

def age_band(a):
    for hi, lab in [(30, "<30"), (35, "30-34"), (40, "35-39"), (45, "40-44"), (50, "45-49"),
                    (55, "50-54"), (60, "55-59"), (65, "60-64")]:
        if a < hi:
            return lab
    return "65+"

def policy_year_band(y):
    return str(y) if y <= 5 else "6-10" if y <= 10 else "11-15" if y <= 15 else "16-20" if y <= 20 else "21+"

def onset_group(a):
    return "<40" if a < 40 else "40-49" if a < 50 else "50-59" if a < 60 else "60+"

def benefit_band(monthly):
    for hi, lab in [(2500, "<2500"), (5000, "2500-4999"), (7500, "5000-7499"), (10000, "7500-9999"),
                    (15000, "10000-14999"), (20000, "15000-19999")]:
        if monthly < hi:
            return lab
    return "20000+"

def sdi_monthly(income):
    wk = income / 52
    return min(SDI_CAP, (0.9 if wk <= SDI_90_LIMIT else 0.7) * wk) * 52 / 12

def _bracket_tax(x, brackets):
    tax, lo = 0.0, 0.0
    for hi, rate in brackets:
        if x <= lo:
            break
        tax += (min(x, hi) - lo) * rate
        lo = hi
    return tax

def income_taxes(gross, california=True):
    """2026 taxes on a year's wages for a single filer with the standard deduction."""
    federal = _bracket_tax(max(0.0, gross - FED_STD), FED_BRACKETS)
    fica = (SS_RATE * min(gross, SS_WAGE_BASE) + MEDICARE_RATE * gross
            + MEDICARE_EXTRA * max(0.0, gross - MEDICARE_EXTRA_OVER))
    state = 0.0
    if california:
        state = max(0.0, _bracket_tax(max(0.0, gross - CA_STD), CA_BRACKETS) - CA_CREDIT) + CA_SDI_RATE * gross
    return federal + fica + state

def take_home(gross, california=True):
    """Yearly take-home pay after 2026 federal, Social Security/Medicare and (optionally) California taxes.
    Outside California no state income tax is assumed."""
    return gross - income_taxes(gross, california)

def default_income_path(age, income):
    """About 15% growth every 5 years until 50, then flat (an assumption; override it)."""
    out, a = {}, math.ceil((age + 1) / 5) * 5
    while a <= 60:
        out[a] = round(income * 1.15 ** (max(0, min(a, 50) - age) / 5) / 5000) * 5000
        a += 5
    return out


# ---------------------------------------------------------------- claim-duration tables
def _glm_rate(M, prof, band):
    g = lambda part, key: M[part].get(key, 0.0)
    return math.exp(M["b0"] + g("dur", band) + g("onset", prof["onset"]) + g("sex", prof["sex"])
                    + g("cls", prof["cls"]) + g("ben", prof["ben"]) + g("diag", prof["diag"]))

def long_claim_table(p: Person, onset, diag, months=600):
    """Monthly (recovery, death) probabilities for claim months 4..months."""
    prof = dict(onset=onset, sex="Female" if p.sex == "f" else "Male", cls=str(p.occ_class),
                ben=benefit_band(p.income * 0.6 / 12), diag=diag)
    bands = {lab: (1 - math.exp(-_glm_rate(GLM["rec"], prof, lab)), 1 - math.exp(-_glm_rate(GLM["dth"], prof, lab)))
             for _, lab in LT_BANDS}
    rec, dth = np.zeros(months + 1), np.zeros(months + 1)
    for m in range(4, months + 1):
        lab = next(l for hi, l in LT_BANDS if m <= hi)
        rec[m], dth[m] = bands[lab]
    return rec, dth


# ---------------------------------------------------------------- simulation
def simulate(p: Person, n=N_LIVES, seed=SEED):
    """Simulate n lives. Returns episode arrays (life, start month, length) and death month per life."""
    rng = np.random.default_rng(seed)
    T = max(12, (65 - p.age) * 12)
    S = SSA[p.sex]
    extra = BMI_RR[p.bmi] / sum(BMI_RR[k] * BMI_MIX[k] for k in BMI_MIX) * (1.5 if p.chronic else 1) * (1.15 if p.hobby else 1)
    mort_rr = ((1.8 if p.smoker else 1) / (0.88 + 0.12 * 1.8) * (1.3 if p.bmi == "obese" else 1) / (0.4 * 1.3 + 0.6)
               * (1.4 if p.chronic else 1))
    base = (INC["b0"] + INC["sex"].get("Female" if p.sex == "f" else "Male", 0) + INC["cls"].get(str(p.occ_class), 0)
            + INC["smoker"].get("Yes" if p.smoker else "No", 0) + INC["ep"]["30"]
            + (INC["state"]["California"] if p.california else 0))
    # monthly hazards (continuous rates) for starting a disability (>30 days) and for death
    hI, hD = np.zeros(T), np.zeros(T)
    for m in range(T):
        a = p.age + m / 12
        annual = math.exp(base + INC["age"].get(age_band(a), 0) + INC["pdur"].get(policy_year_band(m // 12 + 1), 0)) * extra
        hI[m] = annual / 12
        k = min(max(int(a) - 20, 0), 46)
        hD[m] = -math.log(1 - monthly_prob(S["qa"][k] * mort_rr))
    cI, cD = np.r_[0, np.cumsum(hI)], np.r_[0, np.cumsum(hD)]

    def next_event(C, t, e):              # time when cumulative hazard from t grows by e
        target = np.interp(t, np.arange(T + 1), C) + e
        if target >= C[-1]:
            return np.inf
        j = np.searchsorted(C, target) - 1
        return j + (target - C[j]) / (C[j + 1] - C[j])

    s60_30 = math.exp(INC["ep"]["60"] - INC["ep"]["30"])      # share of 30-day claims still open at 60 days
    s90_60 = math.exp(-INC["ep"]["60"])                       # ... at 90 days, given open at 60
    mix = GLM["mix"][p.sex]
    diags, weights = list(mix), np.array(list(mix.values()))
    weights = weights / weights.sum()
    tables = {}

    lives, starts, lengths, death = [], [], [], np.full(n, float(T))
    for i in range(n):
        t = 0.0
        while t < T:
            td = next_event(cD, t, rng.exponential())
            ti = next_event(cI, t, rng.exponential())
            if min(td, ti) >= T:
                break
            if td < ti:
                death[i] = td
                break
            start, died = ti, False
            if rng.random() > s60_30:
                d = 1 + rng.random()
            elif rng.random() > s90_60:
                d = 2 + rng.random()
            else:                                           # lasts past month 3: real claim durations
                og = onset_group(p.age + start / 12)
                dg = diags[rng.choice(len(diags), p=weights)]
                if (og, dg) not in tables:
                    rec, dth = long_claim_table(p, og, dg)
                    surv = np.cumprod(1 - rec - dth)
                    tables[(og, dg)] = (rec, dth, surv)
                rec, dth, surv = tables[(og, dg)]
                u = rng.random()
                m = int(np.searchsorted(-surv[3:], -u)) + 3      # first month where survival drops below u
                if m >= len(surv):
                    d = 1e9
                else:
                    d = m
                    died = rng.random() < dth[m] / (rec[m] + dth[m])
            length = min(d, T - start)
            lives.append(i); starts.append(start); lengths.append(length)
            if died and d <= T - start:
                death[i] = start + d
                break
            t = math.ceil(start + d)
    return dict(T=T, life=np.array(lives, int), start=np.array(starts), length=np.array(lengths), death=death, n=n)


# ---------------------------------------------------------------- money
class Money:
    """Income path, SDI and discounting over the simulation horizon.

    gross: monthly pre-tax income (policy benefits and SDI are set from this)
    inc:   monthly income that a disability takes away; take-home pay when p.after_tax,
           since policy benefits (premiums paid with your own money), SDI and SSDI are tax-free
    """
    def __init__(self, p: Person, T):
        path = {p.age: p.income, **(p.income_path or default_income_path(p.age, p.income))}
        ages, incs = zip(*sorted(path.items()))
        self.T, self.lnv = T, math.log(1 + p.discount_rate / 100) / 12
        mid_age = p.age + (np.arange(T) + 0.5) / 12
        yearly = np.interp(mid_age, ages, incs)
        self.gross = yearly / 12
        self.inc = np.array([take_home(y, p.california) for y in yearly]) / 12 if p.after_tax else self.gross
        self.sdi = np.array([sdi_monthly(y) for y in yearly]) if p.california else np.zeros(T)
        v = np.exp(-self.lnv * (np.arange(T) + 0.5))
        self.CI = np.r_[0, np.cumsum(self.inc * v)]
        self.CS = np.r_[0, np.cumsum(self.sdi * v)]
        self.CR = np.r_[0, np.cumsum(self.gross / (p.income / 12) * v)]
        self.nominal_ratio = float(np.sum(self.gross / (p.income / 12)))

    def span(self, C, a, b):                      # discounted total between months a and b
        a, b = np.clip(a, 0, self.T), np.clip(b, 0, self.T)
        x = np.arange(self.T + 1)
        return np.where(b > a, np.interp(b, x, C) - np.interp(a, x, C), 0.0)

    def ann(self, a, b):                          # discounted months between a and b (flat $1 a month)
        if self.lnv == 0:
            return np.maximum(b - a, 0)
        return np.where(b > a, (np.exp(-self.lnv * a) - np.exp(-self.lnv * b)) / self.lnv, 0.0)

    def at(self, x, arr):
        return arr[np.clip(np.floor(x).astype(int), 0, self.T - 1)]


def baseline(sim, p, M):
    """Loss per life with no private policy: income lost minus SDI (today's dollars; take-home if p.after_tax)."""
    st, d, life = sim["start"], sim["length"], sim["life"]
    lost = M.span(M.CI, st, st + d)
    sdi = np.where(d > 0.23, M.span(M.CS, st + 0.23, st + np.minimum(d, 12)), 0.0)
    n = sim["n"]
    per_life = lambda x: np.bincount(life, weights=x, minlength=n)
    longest = np.zeros(n); np.maximum.at(longest, life, d)
    return dict(loss=per_life(lost) - per_life(sdi), p_any=np.mean(np.bincount(life, minlength=n) > 0),
                p3=np.mean(longest > 3), p12=np.mean(longest > 12),
                exp_lost=per_life(lost).mean(), exp_sdi=per_life(sdi).mean())


def run_policy(sim, p, pol: Policy, base, M):
    st, d, life, n = sim["start"], sim["length"], sim["life"], sim["n"]
    e = pol.elim_months
    bp = 1e9 if pol.benefit_months is None else pol.benefit_months
    claims = d > e
    ben_m = pol.benefit_pct / 100 * (M.at(st, M.gross) if pol.future_increase else p.income / 12)
    b0, b1 = st + e, st + np.minimum(d, e + bp)
    benefit = np.where(claims, M.ann(b0, b1) * ben_m, 0.0)
    if pol.sdi_offset and p.california:
        o1 = np.minimum(b1, st + 12)
        benefit -= np.where(claims & (o1 > b0), M.ann(b0, o1) * np.minimum(ben_m, M.at(st, M.sdi)), 0.0)
    cover = (lambda a, b: M.span(M.CR, a, b)) if pol.future_increase else M.ann
    waived = np.where(claims, cover(st + e, st + d), 0.0)
    ben = np.bincount(life, weights=benefit, minlength=n)
    prem_units = np.maximum(0, cover(np.zeros(n), sim["death"]) - np.bincount(life, weights=waived, minlength=n))
    paid_out = np.bincount(life[claims], minlength=n) > 0
    premium = (ben.mean() / (p.payout_ratio / 100 * prem_units.mean()) if prem_units.mean() > 0 else 0.0) \
        if pol.premium is None else pol.premium
    loss = base["loss"] - ben + premium * prem_units
    months_to_65 = M.nominal_ratio if pol.future_increase else M.T
    return dict(loss=loss, premium=premium, p_claim=paid_out.mean(), exp_benefits=ben.mean(),
                exp_premiums=premium * prem_units.mean(), lifetime_premiums=premium * months_to_65)


def years_below_floor(sim, p, pol, M, floor, ssdi=18_000, ssdi_approval=0.5):
    """Years each life spends with income below `floor` a year (SDI + SSDI if approved + benefits).
    All three are tax-free here, so compare them with a take-home floor."""
    st, d, life, n = sim["start"], sim["length"], sim["life"], sim["n"]
    idx_in_life = np.zeros(len(life), int)
    for k in range(1, len(life)):
        idx_in_life[k] = idx_in_life[k - 1] + 1 if life[k] == life[k - 1] else 0
    h = (((life + 1) * 2654435761 + (2 * idx_in_life + 7) * 40503) % 2**32) / 2**32
    ssdi_on = (h < ssdi_approval) & (d >= 12)
    s_m = M.at(st, M.sdi)
    e = pol.elim_months if pol else 1e9
    bp = (1e9 if pol.benefit_months is None else pol.benefit_months) if pol else 0
    ben_m = (pol.benefit_pct / 100 * (M.at(st, M.gross) if pol.future_increase else p.income / 12)) if pol else 0 * st
    pts = np.stack([np.zeros_like(d), d, np.full_like(d, 0.23), np.full_like(d, 12), np.full_like(d, 5),
                    np.full_like(d, float(e)), np.full_like(d, float(e + bp))], axis=1)
    pts = np.sort(np.minimum(pts, d[:, None]), axis=1)
    below = np.zeros_like(d)
    for j in range(pts.shape[1] - 1):
        a, b = pts[:, j], pts[:, j + 1]
        mid = (a + b) / 2
        inc = np.where((mid >= 0.23) & (mid < 12), s_m, 0) + np.where(ssdi_on & (mid >= 5), ssdi / 12, 0)
        if pol:
            pay = ben_m - (np.minimum(ben_m, s_m) if (pol.sdi_offset and p.california) else 0) * (mid < 12)
            inc = inc + np.where((mid >= e) & (mid < e + bp), pay, 0)
        below += np.where((b > a) & (inc < floor / 12), b - a, 0)
    return np.bincount(life, weights=below, minlength=n) / 12


def run_model(p: Person, policies, sim=None):
    """Everything the dashboard shows, as numbers."""
    sim = sim or simulate(p)
    M = Money(p, sim["T"])
    base = baseline(sim, p, M)
    return dict(sim=sim, money=M, base=base, policies=[run_policy(sim, p, pol, base, M) for pol in policies])


def exceed(values, x):
    """Share of lives with a value above x."""
    s = np.sort(values)
    return 1 - np.searchsorted(s, x, side="right") / len(s)


if __name__ == "__main__":
    import time
    t = time.time()
    p = Person()
    pols = [Policy(elim_months=3), Policy(elim_months=6), Policy(elim_months=12)]
    r = run_model(p, pols)
    print(f"simulated in {time.time()-t:.1f}s; disability >30d {r['base']['p_any']:.0%}, >1y {r['base']['p12']:.1%}")
    for name, x in zip("ABC", r["policies"]):
        print(name, f"premium ${x['premium']:.0f}/mo  pays out {x['p_claim']:.1%}  "
              f"1-in-100 loss ${np.quantile(x['loss'], .99):,.0f}")
    print("no policy 1-in-100 loss", f"${np.quantile(r['base']['loss'], .99):,.0f}")
    print(f"take-home on $95k in California: ${take_home(95_000):,.0f}")
