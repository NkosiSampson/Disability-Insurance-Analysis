# %% [markdown]
# Disability claim termination analysis, SOA IDEC 2006-2015 data
#
# Fits Poisson regression models of how fast disability claims end
# (by recovery or death), using the full dataset hidden in the pivot file.
#
# Needs, in the same folder:
#   read_idec_pivot.py
#   idec-2006-15-ct-pivot.xlsb
# Packages: numpy, pandas, statsmodels, matplotlib  (Colab has all of them)
#
# Run in Colab:  %run fit_termination.py
# Run locally:   python fit_termination.py

# %% 1. Load the full dataset (about 3.3 million rows)
import os
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from read_idec_pivot import load_idec

DATA_FILE = "idec-2006-15-ct-pivot.xlsb"
df = load_idec(DATA_FILE)
print(f"Loaded {len(df):,} rows")

# %% 2. Keep personal disability policies that pay to age 65-70 or for life
df = df[df["type_of_policy"].str.startswith("2 =") # The Soa report says A&S was 94% of policies over 2006–2014. So, filtering the pivot table, we see that refers to type_of_policy == 2
        & df["study_benefit_period_units"].isin(["2", "3"])].copy() # by checking the pivot table with fields "age at onset", "age attatined", and "count exposed" you can see 2 and 3 are to a high age or life.

# Month on claim. "M07" = month 7. "Y06" rows cover a whole year, and their
# exposure is counted in claim-years, so convert to claim-months (x12).
dur = df["duration"].astype(str)
is_year = dur.str.startswith("Y")
num = dur.str[1:].astype(int)
df["month"] = np.where(is_year, (num - 1) * 12 + 6, num)     # midpoint month
df["exposure_months"] = df["count_exposed"] * np.where(is_year, 12, 1)
df["terminations"] = df["death_count"] + df["recovery_count"]

# Grouped explanatory variables
df["dur_band"] = pd.cut(df["month"], [0, 3, 6, 12, 24, 36, 60, 120, 180, 240, 1000],
                        labels=["1-3m", "4-6m", "7-12m", "13-24m", "25-36m", "37-60m",
                                "6-10y", "11-15y", "16-20y", "21y+"])
onset_map = {"<30": "<40", "30-34": "<40", "34-39": "<40", "40-44": "40-49", "44-49": "40-49",
             "50-54": "50-59", "54-59": "50-59", "60-64": "60+", "65-69": "60+", "70-74": "60+"}
df["onset"] = df["age_at_onset"].astype(str).map(onset_map)
df["sex"] = df["gender"].astype(str).str.split("= ").str[1]
df["occ_class"] = df["study_idec_class"].astype(str)
ep = pd.to_numeric(df["study_rounded_elimination_period"].astype(str), errors="coerce")
df["elim"] = pd.cut(ep, [-1, 30, 60, 90, 10000], labels=["<=30d", "60d", "90d", "180d+"])
df["benefit"] = df["monthly_indemnity_amount"].astype(str).str.strip()
df["diagnosis"] = df["study_diagnosis_category_description"].astype(str).str[3:]

# %% 3. Collapse to one row per combination (much faster to model, same answer)
keys = ["dur_band", "onset", "sex", "occ_class", "elim", "benefit", "diagnosis"]
cells = (df.dropna(subset=keys)
           .groupby(keys, observed=True)[["exposure_months", "terminations",
                                          "death_count", "recovery_count"]]
           .sum().reset_index())
cells = cells[cells["exposure_months"] > 0]
print(f"{len(cells):,} cells, {cells.exposure_months.sum()/12:,.0f} claim-years, "
      f"{cells.terminations.sum():,.0f} terminations")

# %% 4. Poisson regression: terminations with log(exposure) as offset
# exp(coefficient) = rate ratio vs. the reference group, holding everything else fixed.
# Counts are weighted (non-integer), so standard errors use a quasi-Poisson scale.
REF = ("C(dur_band, Treatment('4-6m')) + C(onset, Treatment('40-49')) + "
       "C(sex, Treatment('Male')) + C(occ_class, Treatment('1')) + "
       "C(elim, Treatment('90d')) + C(benefit, Treatment('<2500')) + "
       "C(diagnosis, Treatment('Back'))")

def fit(outcome):
    m = smf.glm(f"{outcome} ~ {REF}", data=cells, family=sm.families.Poisson(),
                offset=np.log(cells["exposure_months"])).fit(scale="X2")
    ci = np.exp(m.conf_int())
    out = pd.DataFrame({"rate_ratio": np.exp(m.params), "ci_low": ci[0], "ci_high": ci[1],
                        "p_value": m.pvalues})
    out.index = (out.index.str.replace(r"C\((\w+), Treatment\('[^']*'\)\)\[T\.", r"\1: ", regex=True)
                          .str.rstrip("]"))
    return m, out.round(3)

model_all, rr_all = fit("terminations")
model_rec, rr_rec = fit("recovery_count")
model_dth, rr_dth = fit("death_count")

pd.set_option("display.max_rows", 100, "display.max_columns", 10, "display.width", 140)
print("\nTERMINATIONS (recovery + death): rate ratios")
print(rr_all)
print("\nRECOVERY ONLY: rate ratios")
print(rr_rec)
print("\nDEATH ONLY: rate ratios")
print(rr_dth)
rr_all.to_csv("rate_ratios_terminations.csv")
rr_rec.to_csv("rate_ratios_recovery.csv")
rr_dth.to_csv("rate_ratios_death.csv")


def save_coefficients(model, path):
    """Write a fitted model's coefficients as a readable table; disability_model.py reads these files."""
    rows, seen = [("Intercept", "", model.params["Intercept"])], set()
    for name, b in model.params.items():
        m = re.match(r"C\((\w+), Treatment\('([^']*)'\)\)\[T\.(.*)\]$", name)
        if not m:
            continue
        var, ref, level = m.groups()
        if var not in seen:                     # the reference level has coefficient 0
            seen.add(var); rows.append((var, ref, 0.0))
        rows.append((var, level, b))
    out = pd.DataFrame(rows, columns=["variable", "level", "coefficient"])
    out["coefficient"] = out["coefficient"].round(5)
    out["rate_ratio"] = np.where(out["variable"] == "Intercept", np.nan, np.exp(out["coefficient"]).round(4))
    os.makedirs("model_inputs", exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Wrote {path}")

save_coefficients(model_rec, "model_inputs/recovery_coefficients.csv")
save_coefficients(model_dth, "model_inputs/death_coefficients.csv")


# %% 5. Predicted claim survival curves for example profiles
# Chance a claim is still open, starting from month 3 (end of a 90-day wait).
bands = cells["dur_band"].cat.categories
band_months = [(1, 3), (4, 6), (7, 12), (13, 24), (25, 36), (37, 60),
               (61, 120), (121, 180), (181, 240), (241, 360)]

def survival_curve(profile, model=model_all, start=3, end=360):
    rows = pd.DataFrame([{**profile, "dur_band": b} for b in bands])
    rows["dur_band"] = pd.Categorical(rows["dur_band"], categories=bands)
    monthly = model.predict(rows, offset=np.zeros(len(rows)))   # rate per claim-month
    hazard = np.zeros(end + 1)
    for (lo, hi), h in zip(band_months, monthly):
        hazard[lo:hi + 1] = h
    months = np.arange(start, end + 1)
    surv = np.exp(-np.cumsum(hazard[start + 1:end + 1]))
    return months, np.r_[1.0, surv]

base = dict(onset="<40", sex="Male", occ_class="1", elim="90d", benefit="5000-7499", diagnosis="Back")
profiles = {
    "Male, class 1, onset <40, back": base,
    "Female, same otherwise": {**base, "sex": "Female"},
    "Male, onset 50-59": {**base, "onset": "50-59"},
    "Male, cancer diagnosis": {**base, "diagnosis": "Cancer"},
    "Male, mental & nervous": {**base, "diagnosis": "Mental & Nervous"},
}
plt.figure(figsize=(9, 5))
for label, prof in profiles.items():
    m, s = survival_curve(prof)
    plt.plot(m / 12, s, label=label)
plt.xlabel("Years since disability began")
plt.ylabel("Chance the claim is still open")
plt.title("Predicted disability claim survival (SOA IDEC 2006-2015, Poisson model)")
plt.ylim(0, 1); plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig("claim_survival_curves.png", dpi=150)
plt.show()

for label, prof in profiles.items():
    m, s = survival_curve(prof)
    at = lambda y: s[m.searchsorted(y * 12)]
    print(f"{label:34s}  open at 1y: {at(1):.0%}   5y: {at(5):.0%}   10y: {at(10):.0%}   20y: {at(20):.0%}")
