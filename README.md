# Choosing Disability Insurance with Actuarial Data

Which disability insurance policy, if any, is worth buying given your own risk factors? This project answers that with a simulation grounded in real insurance data rather than rules of thumb.

I decoded the raw records hidden inside the Society of Actuaries' 2006–2015 individual disability studies (about 7 million records in total) and fitted Poisson regression models to them: one for how often disability claims start (25.4 million policy-years, 148,000 claims) and two for how long they last, through recovery or death (762,000 claim-years). Those models drive a simulation of 50,000 working lives, month by month from your current age to 65, including California SDI, Social Security disability, and your expected income growth.

For each policy you set up (waiting period, benefit period, benefit percentage, premium, and a future increase option), the dashboard shows the chance it pays out, expected cost versus expected benefits, worst-case losses, and how likely a disability is to leave you below the income you need to support yourself, and for how long.

The result is not a recommendation. Every policy pays back about 60 cents per premium dollar on average, so the tool lays out what that cost buys in protection and leaves the decision to you.

## Files

| File | What it is |
| --- | --- |
| `app.py` | The interactive dashboard (Streamlit) |
| `disability_model.py` | The simulation model, with the fitted SOA coefficients built in |
| `requirements.txt` | Python packages the app needs |

## Run it yourself

```
pip install -r requirements.txt
streamlit run app.py
```

## Data sources

- Society of Actuaries, IDEC claim incidence study, 2006–2015
- Society of Actuaries, IDEC claim termination study, 2006–2015
- Social Security Administration, Actuarial Note 2026.6
- California EDD, 2026 State Disability Insurance rules

Built in Python with numpy, pandas, statsmodels, and Streamlit. This is an educational model, not financial advice.
