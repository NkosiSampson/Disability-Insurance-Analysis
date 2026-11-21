"""
Disability insurance policy model (Python version of the dashboard's engine).

Simulates working lives month by month from your age to 65, using models fitted to
Society of Actuaries (SOA) individual disability data, then compares insurance policies.

    from disability_model import Person, Policy, run_model
    res = run_model(Person(age=23, income=95000), [Policy(elim_months=3), Policy(elim_months=12)])

Data (see the notebooks for how the models were fitted):
  INC  Poisson model of how often claims start (SOA IDEC claim incidence study, 2006-2015)
  GLM  Poisson models of recovery and death while disabled (SOA IDEC claim termination study)
  SSA  healthy-worker mortality by age and sex (SSA Actuarial Note 2026.6)
  TAX  2026 federal, Social Security/Medicare and California income taxes (single filer),
       used to turn income into take-home pay
Only numpy is required.
"""
from dataclasses import dataclass, field
from typing import Optional
import json
import math
import numpy as np

DATA = json.loads(r'''{"SSA":{"m":{"inc":[0.001789,0.001675,0.001557,0.001695,0.001908,0.001538,0.001164,0.001335,0.001462,0.001611,0.001712,0.001766,0.001849,0.001947,0.002041,0.002202,0.002379,0.002506,0.002641,0.00281,0.002956,0.003108,0.003309,0.003515,0.00374,0.004022,0.004328,0.00461,0.004846,0.005052,0.006395,0.007995,0.008069,0.007959,0.008289,0.010759,0.013626,0.013872,0.014003,0.014915,0.015766,0.016771,0.017626,0.017217,0.014791,0.011738,0.008757],"qa":[0.001106,0.001167,0.00121,0.001244,0.001281,0.001339,0.00142,0.001492,0.00156,0.001631,0.001698,0.001771,0.001849,0.001934,0.002017,0.002084,0.002134,0.002176,0.002211,0.002241,0.002269,0.00229,0.002323,0.002333,0.002336,0.002345,0.002352,0.002358,0.002407,0.002511,0.002686,0.00286,0.002981,0.003093,0.003214,0.003482,0.003716,0.003943,0.004213,0.004467,0.004815,0.005157,0.005426,0.005871,0.006379,0.007111,0.007786],"qd":[0,0.01293,0.00965,0.00873,0.00871,0.00806,0.00729,0.00793,0.00865,0.00903,0.00986,0.01035,0.01092,0.01128,0.0116,0.0122,0.01283,0.01323,0.01366,0.01427,0.01495,0.01568,0.01594,0.0167,0.01737,0.01764,0.01781,0.01829,0.01875,0.01938,0.01984,0.02024,0.02092,0.02164,0.02247,0.02203,0.02215,0.02247,0.02288,0.02366,0.02388,0.02431,0.02535,0.02576,0.02652,0.02708,0.02746],"rec":[0,0.00337,0.00322,0.00406,0.00932,0.01925,0.02219,0.02256,0.02341,0.02262,0.02528,0.02763,0.02672,0.0254,0.02428,0.02461,0.02457,0.02332,0.02231,0.02222,0.02114,0.02002,0.01877,0.01715,0.01657,0.01646,0.01537,0.01403,0.01299,0.01152,0.01104,0.01015,0.00892,0.00805,0.00687,0.00645,0.00619,0.00533,0.00466,0.00414,0.00431,0.00453,0.00398,0.00326,0.00283,0.00235,0.00193]},"f":{"inc":[0.00134,0.001224,0.001098,0.001235,0.001454,0.001249,0.001058,0.001244,0.001387,0.001555,0.001708,0.001818,0.001949,0.002114,0.002291,0.00246,0.002619,0.002783,0.002952,0.003145,0.003503,0.003877,0.004089,0.004323,0.004576,0.004841,0.005131,0.005397,0.005626,0.005817,0.007323,0.009075,0.009029,0.008788,0.009029,0.011262,0.013807,0.013902,0.013873,0.014494,0.014249,0.014133,0.0147,0.01432,0.01235,0.00992,0.007496],"qa":[0.000403,0.000424,0.000444,0.000459,0.000474,0.000493,0.000512,0.000545,0.000584,0.00063,0.000671,0.000715,0.000764,0.000806,0.000848,0.000876,0.000895,0.000918,0.000956,0.000986,0.000994,0.001007,0.001018,0.001057,0.001083,0.001111,0.001129,0.001154,0.001171,0.001233,0.001319,0.001399,0.001457,0.001549,0.001651,0.001808,0.002014,0.002136,0.002288,0.002493,0.002671,0.002823,0.002953,0.003107,0.003456,0.003969,0.004335],"qd":[0,0.01049,0.00788,0.00777,0.00772,0.00714,0.00744,0.00728,0.00767,0.00829,0.00973,0.0106,0.01066,0.01077,0.01079,0.0111,0.01184,0.01212,0.01169,0.01153,0.01207,0.01233,0.01281,0.01277,0.01306,0.01315,0.01329,0.01319,0.01352,0.01356,0.01375,0.01392,0.01418,0.01422,0.01453,0.01465,0.01444,0.01492,0.01529,0.0154,0.01572,0.0162,0.01686,0.01757,0.01794,0.01833,0.01909],"rec":[0,0.003,0.00315,0.00388,0.00689,0.01638,0.0197,0.01924,0.01886,0.01795,0.02172,0.02432,0.02294,0.02225,0.02134,0.02146,0.02194,0.02103,0.01997,0.01886,0.01849,0.01814,0.01741,0.01647,0.0157,0.01541,0.01461,0.01389,0.01285,0.01176,0.01094,0.00994,0.00898,0.00802,0.00697,0.00639,0.00591,0.00523,0.0047,0.00421,0.00404,0.00387,0.00332,0.0028,0.00242,0.00199,0.00164]}},"GLM":{"rec":{"b0":-2.9493,"dur":{"4-6m":0,"1-3m":0.23935,"7-12m":-0.32117,"13-24m":-1.4507,"25-36m":-2.33263,"37-60m":-3.24815,"6-10y":-4.31551,"11-15y":-5.17721,"16-20y":-5.73734,"21y+":-6.50727},"onset":{"40-49":0,"50-59":-0.31375,"60+":-0.53822,"<40":0.26617},"sex":{"Male":0,"Female":0.04927},"cls":{"1":0,"2":0.0616,"3":0.07713,"4":0.02359,"M":0.10025},"ben":{"<2500":0,"10000-14999":-0.39862,"15000-19999":-0.68563,"20000+":-1.00572,"2500-4999":-0.11375,"5000-7499":-0.21995,"7500-9999":-0.3237},"diag":{"Back":0,"Alcohol & Drug":0.65242,"Cancer":0.19933,"Circulatory":0.14543,"Diabetes":-1.26946,"Diagnosis Not Provided":0.75666,"Digestive":0.56149,"Ill-defined and Misc Conditions":0.23382,"Infectious Diseases":0.15752,"Injury Other Than Back":0.90825,"Maternity":1.76458,"Mental & Nervous":-0.12095,"Nervous System":-0.94638,"Other":0.08873,"Other Musculoskeletal":0.36501,"Respiratory":-0.11554}},"dth":{"b0":-6.66681,"dur":{"4-6m":0,"1-3m":-0.34534,"7-12m":0.11626,"13-24m":0.05377,"25-36m":-0.24965,"37-60m":-0.61793,"6-10y":-0.84358,"11-15y":-0.74995,"16-20y":-0.47841,"21y+":0.11178},"onset":{"40-49":0,"50-59":0.30731,"60+":0.45591,"<40":-0.43716},"sex":{"Male":0,"Female":-0.46113},"cls":{"1":0,"2":-0.14197,"3":0.14295,"4":-0.48626,"M":-0.22835},"ben":{"<2500":0,"10000-14999":-0.61063,"15000-19999":-0.69934,"20000+":-0.86767,"2500-4999":-0.22765,"5000-7499":-0.37514,"7500-9999":-0.44332},"diag":{"Back":0,"Alcohol & Drug":1.02615,"Cancer":2.94893,"Circulatory":0.80816,"Diabetes":1.13123,"Diagnosis Not Provided":1.22224,"Digestive":1.22106,"Ill-defined and Misc Conditions":0.80911,"Infectious Diseases":0.97492,"Injury Other Than Back":0.13915,"Maternity":0.39453,"Mental & Nervous":0.45383,"Nervous System":0.94257,"Other":0.61108,"Other Musculoskeletal":-0.25005,"Respiratory":1.18899}},"mix":{"m":{"Alcohol & Drug":0.0245,"Back":0.1234,"Cancer":0.1666,"Circulatory":0.094,"Diabetes":0.0041,"Digestive":0.0227,"Ill-defined and Misc Conditions":0.0156,"Infectious Diseases":0.0116,"Injury Other Than Back":0.1191,"Maternity":0.0006,"Mental & Nervous":0.1034,"Nervous System":0.0997,"Other":0.0691,"Other Musculoskeletal":0.1344,"Respiratory":0.0112},"f":{"Alcohol & Drug":0.0093,"Back":0.1062,"Cancer":0.1972,"Circulatory":0.0403,"Diabetes":0.0018,"Digestive":0.019,"Ill-defined and Misc Conditions":0.0166,"Infectious Diseases":0.0101,"Injury Other Than Back":0.1083,"Maternity":0.0848,"Mental & Nervous":0.0916,"Nervous System":0.0828,"Other":0.0568,"Other Musculoskeletal":0.1639,"Respiratory":0.0113}}},"INC":{"b0":-6.24753,"age":{"40-44":0,"30-34":-0.26976,"35-39":-0.16973,"45-49":0.27078,"50-54":0.6127,"55-59":0.98359,"60-64":1.23183,"65+":1.15109,"<30":-0.62276},"sex":{"Male":0,"Female":0.37311},"cls":{"1":0,"2":0.2319,"3":0.66962,"4":0.75345,"M":0.43987},"smoker":{"No":0,"Unknown":0.11691,"Yes":0.31663},"ep":{"90":0,"180+":-0.49645,"30":0.96224,"60":0.41596,"<30":1.06247},"bp":{"ToAge":0,"Lifetime":0.1935,"Short":0.06834},"state":{"Others":0,"California":0.23579,"Florida":0.11414},"pdur":{"6-10":0,"1":-0.04479,"11-15":0.03467,"16-20":0.05826,"2":-0.07916,"21+":-0.0522,"3":-0.01841,"4":0.0358,"5":0.01936},"uw":{"1":0,"2":0.07386,"4":0.28937},"mkt":{"1":0,"2":-0.04749,"3":-0.18176,"4":-0.1642,"5":-0.01322,"X":-0.32025}}}''')
SSA, GLM, INC = DATA["SSA"], DATA["GLM"], DATA["INC"]

N_LIVES = 50_000
SEED = 20260926
SDI_CAP, SDI_90_LIMIT = 1765.0, 1252.30            # 2026 California SDI weekly cap and 90% threshold
BMI_RR = {"under": 1.20, "normal": 1.0, "over": 1.13, "obese": 1.52}
BMI_MIX = {"under": 0.02, "normal": 0.27, "over": 0.31, "obese": 0.40}
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
