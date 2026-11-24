# Choosing Disability Insurance with Actuarial Data

Which disability insurance policy, if any, is worth buying given your own risk factors? This project answers that with a simulation grounded in real insurance data rather than rules of thumb.

I decoded the raw records hidden inside the Society of Actuaries' 2006–2015 individual disability studies (about 7 million records in total) and fitted Poisson regression models to them: one for how often disability claims start (25.4 million policy-years, 148,000 claims) and two for how long they last, through recovery or death (762,000 claim-years). Those models drive a simulation of 50,000 working lives, month by month from your current age to 65, including California SDI, Social Security disability, taxes, and your expected income growth.

For each policy you set up (waiting period, benefit period, benefit percentage, premium, and a future increase option), the dashboard shows the chance it pays out, expected cost versus expected benefits, worst-case losses, and how likely a disability is to leave you below the income you need to support yourself, and for how long.

Losses are counted in take-home pay by default. A disability takes away your pay after taxes, while policy benefits (when you pay the premiums yourself), SDI and SSDI are tax-free, so a benefit of 60% of your pre-tax income replaces roughly 80% of what you actually take home. A switch in the app shows everything before taxes instead.

The result is not a recommendation. Every policy pays back about 60 cents per premium dollar on average, so the tool lays out what that cost buys in protection and leaves the decision to you.

## Files

| File | What it is |
| --- | --- |
| `app.py` | The interactive dashboard (Streamlit) |
| `disability_model.py` | The simulation model; reads the fitted SOA coefficients from `model_inputs/` |
| `requirements.txt` | Python packages the app needs |
| `read_idec_pivot.py` | Decodes the raw records hidden inside the SOA pivot files |
| `fit_incidence.py` | Fits the claim incidence model (how often claims start) |
| `fit_termination.py` | Fits the claim termination models (recovery and death) |
| `model_inputs/` | The fitted model coefficients and tables the dashboard reads (written by the fit scripts) |

## Run it yourself

```
pip install -r requirements.txt
streamlit run app.py
```

The app does not need the raw data. The fitted coefficients are already stored in `model_inputs/`. You only need the data below if you want to refit the models and check the numbers yourself.

## Data: where it comes from and how to get it

### 1. SOA claim incidence data (how often claims start)

- Study page: [Analysis of Claim Incidence Experience from 2006 to 2014](https://www.soa.org/resources/experience-studies/2019/claim-incidence-report/)
- Data file (about 315 MB): [idec-incidence-table.xlsb](https://cdn-files.soa.org/web/research/idec-incidence-table.xlsb)
- Report: [IDEC 2006–2014 Claim Incidence Report (PDF)](https://www.soa.org/globalassets/assets/files/resources/experience-studies/2019/2006-14-idec-incidence-report.pdf)

### 2. SOA claim termination data (how long claims last)

- Study page: [Analysis of Claim Termination Experience from 2006 to 2014](https://www.soa.org/resources/experience-studies/2021/analysis-claim-termination/)
- Data file (about 300 MB): [idec-2006-15-ct-pivot.xlsb](https://www.soa.org/globalassets/assets/files/resources/experience-studies/2022/idec-2006-15-ct-pivot.xlsb)
- Report: [IDEC 2006–2014 Claim Termination Report (PDF)](https://www.soa.org/globalassets/assets/files/resources/experience-studies/2021/analysis-claim-termination-report.pdf)

The SOA study pages describe 2006–2014, but the termination file's name says 2006–15, so this project refers to the data as 2006–2015.

Both files are free to download, but they are too large to store on GitHub, so they are not in this repo.

**Why a special reader is needed:** opening these files in Excel, pandas or pyxlsb only shows a summary pivot table. The individual records are stored inside the file's pivot cache, a binary part of the `.xlsb` file. `read_idec_pivot.py` reads that part directly and returns every record as a pandas DataFrame.

**Note:** this is data on people who bought individual disability insurance. It describes insured, mostly professional workers, not the general population.

### 3. Social Security mortality and disability rates

- [SSA Actuarial Note 2026.6 (PDF)](https://www.ssa.gov/oact/NOTES/ran6/an2026-6.pdf): death rates for healthy workers by age and sex, used for the chance of dying before 65 while not disabled.

### 4. California State Disability Insurance (SDI)

- [EDD: Disability Insurance benefit payment amounts](https://edd.ca.gov/en/Disability/Calculating_DI_Benefit_Payment_Amounts)
- [EDD: 2026 maximum weekly benefit amount (PDF)](https://edd.ca.gov/siteassets/files/about_edd/quick-stats/max-weekly-benefit-amount-2026.pdf)

The model uses the 2026 rules: 90% of weekly wages up to $1,252.30, otherwise 70%, capped at $1,765 a week, after a 7-day wait, for up to 52 weeks.

### 5. 2026 taxes (for take-home pay)

- [IRS: 2026 tax brackets and standard deduction](https://www.irs.gov/newsroom/irs-releases-tax-inflation-adjustments-for-tax-year-2026-including-amendments-from-the-one-big-beautiful-bill)
- [California FTB: 2026 tax rate schedule and standard deduction](https://www.ftb.ca.gov/about-ftb/newsroom/tax-news/2026/10.html)
- [CalChamber: 2026 Social Security wage base and California SDI payroll rate](https://hrcalifornia.calchamber.com/articles/2026-social-security-taxable-wage-base-california-sdi-withholding-rate-increase)

The model assumes a single filer taking the standard deduction: federal income tax, Social Security (6.2% up to $184,500), Medicare (1.45%, plus 0.9% over $200,000), California income tax and the 1.3% California SDI payroll tax. Outside California, no state income tax is assumed. On $95,000 in California this gives about $70,000 of take-home pay.

### 6. U.S. adult body mass index (BMI) distribution (CDC)

- [CDC Health E-Stat 111: Prevalence of Overweight, Obesity, and Severe Obesity Among Adults Age 20 and Older](https://www.cdc.gov/nchs/data/hestat/hestat111.htm): in August 2021–August 2023, 40.3% of U.S. adults had obesity (BMI 30 or higher) and another 31.7% were overweight (BMI 25 to 29.9), age adjusted. The same figures are in the last row of Table 1.
- [CDC Health E-Stat 119: Prevalence of Underweight Among Children, Adolescents, and Adults](https://cdc.gov/nchs/data/hestat/hestat119.htm): in the same period, 1.6% of adults were underweight (BMI under 18.5), age adjusted. Also in the last row of Table 2.
- Normal weight (BMI 18.5 to 24.9) is not published as its own figure; it is the remainder, 100% − 40.3% − 31.7% − 1.6% = 26.4%.

The model stores these, rounded, as `BMI_MIX` in `disability_model.py` (underweight 2%, normal 27%, overweight 31%, obese 40%). The SOA claim rates already reflect a mix of body weights, so the BMI risk multipliers are divided by their average over this mix: an average-BMI person's rate stays at the SOA rate, and only people above or below average move. The obese share (40%) is also used the same way for the death-rate adjustment. The BMI multipliers themselves are assumptions, not fitted from this data.

## Refit the models from the raw data

1. Download the two `.xlsb` files above and put them in the same folder as the scripts.
2. Install the packages: `pip install numpy pandas statsmodels matplotlib`
3. Check that the data loads:
   ```
   python read_idec_pivot.py idec-incidence-table.xlsb
   python read_idec_pivot.py idec-2006-15-ct-pivot.xlsb
   ```
   Each command writes a CSV with every record and prints totals you can compare with the SOA report.
4. Fit the models:
   ```
   python fit_incidence.py
   python fit_termination.py
   ```
   Each script prints the rate ratios with confidence intervals, saves a chart, and writes its coefficients to `model_inputs/`, which the dashboard reads. So a refit updates the dashboard automatically.

You can also use the reader in your own code:

```python
from read_idec_pivot import load_idec
df = load_idec("idec-incidence-table.xlsb")
```

Each file needs several GB of memory to load in full, so use a machine or Colab runtime with plenty of RAM. In Google Colab, upload the `.xlsb` file and the scripts through the Files panel, then run `%run fit_incidence.py`.

Built in Python with numpy, pandas, statsmodels, and Streamlit. This is an educational model, not financial advice.
