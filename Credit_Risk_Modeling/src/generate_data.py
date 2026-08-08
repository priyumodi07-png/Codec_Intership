"""
Synthetic credit application data generator.

Mimics the structure and feature relationships of well-known public
credit-risk datasets (UCI German Credit Data / Kaggle Home Credit).

Swap-in instructions for the REAL dataset:
  1. Download "Home Credit Default Risk" from Kaggle:
     https://www.kaggle.com/c/home-credit-default-risk
  2. Place application_train.csv in data/raw/
  3. In src/train.py, change `load_data()` to read that CSV instead of
     calling generate_synthetic_data(). Column names are intentionally
     aligned (SK_ID_CURR, TARGET, AMT_INCOME_TOTAL, etc.) to minimize
     the diff.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_SAMPLES = 12000


def generate_synthetic_data(n=N_SAMPLES, seed=42):
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 70, n)
    employment_years = np.clip(rng.exponential(5, n), 0, 40)
    income = np.clip(rng.normal(45000, 20000, n) + employment_years * 500, 8000, None)
    credit_amount = np.clip(rng.normal(15000, 9000, n), 500, 80000)
    duration_months = rng.choice([6, 12, 18, 24, 36, 48, 60], n,
                                  p=[0.05, 0.15, 0.15, 0.25, 0.2, 0.12, 0.08])

    checking_status = rng.choice(
        ["no_account", "negative", "0_to_200", "200_plus"], n,
        p=[0.35, 0.2, 0.3, 0.15]
    )
    savings_status = rng.choice(
        ["unknown", "low", "medium", "high"], n,
        p=[0.3, 0.4, 0.2, 0.1]
    )
    purpose = rng.choice(
        ["car", "furniture", "electronics", "business", "education", "renovation"],
        n, p=[0.25, 0.2, 0.15, 0.15, 0.1, 0.15]
    )
    housing = rng.choice(["own", "rent", "with_parents"], n, p=[0.55, 0.35, 0.1])
    num_dependents = rng.poisson(0.8, n)
    existing_credits = rng.integers(0, 5, n)
    delinquencies_2yr = rng.poisson(0.3, n)

    debt_to_income = credit_amount / (income + 1)
    credit_utilization = np.clip(rng.beta(2, 5, n) + delinquencies_2yr * 0.05, 0, 1)

    # --- latent default-risk score drives the target (logistic process) ---
    checking_risk = pd.Series(checking_status).map(
        {"no_account": 0.3, "negative": 1.2, "0_to_200": 0.1, "200_plus": -0.4}
    ).values
    savings_risk = pd.Series(savings_status).map(
        {"unknown": 0.3, "low": 0.5, "medium": -0.1, "high": -0.6}
    ).values

    logit = (
        -3.0
        + 1.8 * debt_to_income
        + 1.4 * credit_utilization
        + 0.35 * delinquencies_2yr
        + checking_risk
        + savings_risk
        - 0.015 * (age - 21)
        - 0.05 * employment_years
        + 0.01 * (duration_months / 12)
        - 0.3 * (housing == "own").astype(float)
        + rng.normal(0, 0.6, n)
    )
    prob_default = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, prob_default)

    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(100001, 100001 + n),
        "TARGET": target,
        "AGE": age,
        "AMT_INCOME_TOTAL": income.round(2),
        "AMT_CREDIT": credit_amount.round(2),
        "DURATION_MONTHS": duration_months,
        "EMPLOYMENT_YEARS": employment_years.round(1),
        "CHECKING_STATUS": checking_status,
        "SAVINGS_STATUS": savings_status,
        "PURPOSE": purpose,
        "HOUSING": housing,
        "NUM_DEPENDENTS": num_dependents,
        "EXISTING_CREDITS": existing_credits,
        "DELINQUENCIES_2YR": delinquencies_2yr,
        "DEBT_TO_INCOME": debt_to_income.round(4),
        "CREDIT_UTILIZATION": credit_utilization.round(4),
    })
    return df


if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("data/application_data.csv", index=False)
    print(f"Generated {len(df)} rows -> data/application_data.csv")
    print(f"Default rate: {df['TARGET'].mean():.2%}")
