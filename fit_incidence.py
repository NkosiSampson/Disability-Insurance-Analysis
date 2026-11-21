# %% [markdown]
# Disability claim incidence analysis, SOA IDEC 2006-2015 data
#
# Fits a Poisson regression model of how often new disability claims start,
# using the full dataset hidden in the pivot file.
#
# Needs, in the same folder:
#   read_idec_pivot.py
#   idec-incidence-table.xlsb
# Packages: numpy, pandas, statsmodels, matplotlib  (Colab has all of them)
#
# Run in Colab:  %run fit_incidence.py
# Run locally:   python fit_incidence.py

# %% 1. Load the full incidence dataset (about 4 million rows)
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")
from read_idec_pivot import load_idec

DATA_FILE = "idec-incidence-table.xlsb"
df = load_idec(DATA_FILE)
print(f"Loaded {len(df):,} rows")
print(f"Policy-years: {df.exposure_by_count.sum():,.0f}   New claims: {df.disability_count.sum():,.0f}")


# %% 2. Keep personal disability (accident & sickness) policies and tidy the variables
df = df[df["contract_type"].astype(str).str.startswith("2 ")].copy()
for c in ["attained_age_band", "gender", "idec_occupation_class", "smoking_status",
          "sickness_elimination_period", "sickness_benefit_period", "state_of_issue",
          "duration_band", "underwriting_type", "marketing_source"]:
    df[c] = df[c].astype(str).str.strip()
df = df[(df.duration_band != "Invalid/Missing") & (df.sickness_benefit_period != "Invalid/Missing")]

df["age"] = df.attained_age_band.replace({"Under 30": "<30", "65 +": "65+"})
df["sex"] = df.gender.str.split("= ").str[1]
df["occ_class"] = df.idec_occupation_class
df["smoker"] = df.smoking_status.map({"1 = Non-Smoker": "No", "2 = Smoker or Tobacco user": "Yes", "0 = Unknown": "Unknown"})
df["elim"] = df.sickness_elimination_period.replace({"Under 30 days": "<30", "30 days": "30", "60 days": "60",
                                                     "90 days": "90", "180+ days": "180+"})
df["benefit_period"] = df.sickness_benefit_period.replace({"To XX": "ToAge", "Short Term": "Short", "Lifetime": "Lifetime"})
df["state"] = df.state_of_issue
df["policy_year"] = df.duration_band
df["underwriting"] = df.underwriting_type.str[0]      # 1 full, 2 guaranteed issue, 4 insurability option
df["market"] = df.marketing_source.str[0].replace({"I": "Missing"})   # 1 individual, 2-4 employer, 5 association


# %% 3. Collapse to one row per combination (same answer, much faster)
keys = ["age", "sex", "occ_class", "smoker", "elim", "benefit_period", "state", "policy_year", "underwriting", "market"]
cells = (df.groupby(keys, observed=True)[["exposure_by_count", "disability_count"]]
           .sum().reset_index())
cells = cells[cells.exposure_by_count > 0]
print(f"{len(cells):,} cells, {cells.exposure_by_count.sum():,.0f} policy-years, {cells.disability_count.sum():,.0f} claims")

# Raw claim rates by age and sex, for a quick look
raw = cells.groupby(["age", "sex"])[["exposure_by_count", "disability_count"]].sum()
raw["claims_per_1000"] = 1000 * raw.disability_count / raw.exposure_by_count
print(raw["claims_per_1000"].unstack().round(2))


# %% 4. Poisson regression of new claims, with log(policy-years) as offset
# exp(coefficient) = rate ratio vs. the reference group, holding everything else fixed.
formula = ("disability_count ~ C(age, Treatment('40-44')) + C(sex, Treatment('Male')) + "
           "C(occ_class, Treatment('1')) + C(smoker, Treatment('No')) + C(elim, Treatment('90')) + "
           "C(benefit_period, Treatment('ToAge')) + C(state, Treatment('Others')) + "
           "C(policy_year, Treatment('6-10')) + C(underwriting, Treatment('1')) + C(market, Treatment('1'))")
model = smf.glm(formula, data=cells, family=sm.families.Poisson(),
                offset=np.log(cells.exposure_by_count)).fit(scale="X2")

ci = np.exp(model.conf_int())
rr = pd.DataFrame({"rate_ratio": np.exp(model.params), "ci_low": ci[0], "ci_high": ci[1],
                   "p_value": model.pvalues}).round(3)
rr.index = rr.index.str.replace(r"C\((\w+), Treatment\('[^']*'\)\)\[T\.", r"\1: ", regex=True).str.rstrip("]")
pd.set_option("display.max_rows", 100, "display.max_columns", 10, "display.width", 140)
print(rr)
print(f"\nReference annual claim rate: {np.exp(model.params['Intercept']):.4%} "
      "(age 40-44, male, class 1, non-smoker, 90-day wait, benefits to 65-70, not CA/FL, "
      "policy years 6-10, fully underwritten, individual market)")
rr.to_csv("incidence_rate_ratios.csv")


# %% 5. Predicted annual claim rates by age for example profiles
ages = ["<30", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64"]
base = dict(sex="Male", occ_class="1", smoker="No", elim="90", benefit_period="ToAge",
            state="California", policy_year="6-10", underwriting="1", market="1")
profiles = {
    "Male, class 1, CA, 90-day": base,
    "Female, same otherwise": {**base, "sex": "Female"},
    "Male, heavy manual (class 4)": {**base, "occ_class": "4"},
    "Male, smoker": {**base, "smoker": "Yes"},
    "Male, 30-day wait": {**base, "elim": "30"},
}
plt.figure(figsize=(9, 5))
for label, prof in profiles.items():
    rows = pd.DataFrame([{**prof, "age": a} for a in ages])
    rate = model.predict(rows, offset=np.zeros(len(rows)))
    plt.plot(ages, 1000 * rate, marker="o", label=label)
plt.ylabel("New claims per 1,000 policies per year")
plt.xlabel("Age")
plt.title("Predicted disability claim rates (SOA IDEC incidence data, Poisson model)")
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig("incidence_by_age.png", dpi=150)
plt.show()
