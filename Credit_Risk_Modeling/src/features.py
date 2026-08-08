"""
Feature engineering: Weight of Evidence (WOE) binning and Information Value (IV).

WOE/IV is the industry-standard transformation in credit scoring because it:
  1. Turns any variable (continuous or categorical) into a monotonic,
     linear-in-log-odds feature -> ideal for logistic regression scorecards.
  2. Naturally handles missing values (they just become their own bin).
  3. Produces IV, a single number ranking each feature's predictive power,
     which is how risk teams decide what goes into the model.

IV interpretation (standard industry thresholds):
  < 0.02            : not predictive, drop
  0.02 - 0.1         : weak
  0.1  - 0.3         : medium
  0.3  - 0.5         : strong
  > 0.5              : suspiciously strong -> check for leakage
"""

import numpy as np
import pandas as pd

EPS = 0.5  # Laplace smoothing to avoid divide-by-zero on rare bins


def _woe_iv_table(bin_col: pd.Series, target: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"bin": bin_col, "target": target})
    grp = df.groupby("bin", observed=True)["target"].agg(["count", "sum"])
    grp.columns = ["total", "bad"]
    grp["good"] = grp["total"] - grp["bad"]

    total_bad = grp["bad"].sum()
    total_good = grp["good"].sum()

    grp["dist_bad"] = (grp["bad"] + EPS) / (total_bad + EPS * len(grp))
    grp["dist_good"] = (grp["good"] + EPS) / (total_good + EPS * len(grp))
    grp["woe"] = np.log(grp["dist_good"] / grp["dist_bad"])
    grp["iv"] = (grp["dist_good"] - grp["dist_bad"]) * grp["woe"]
    return grp


def fit_woe_numeric(series: pd.Series, target: pd.Series, bins: int = 5):
    """Quantile-bin a numeric series, return (bin_edges, woe_table)."""
    edges = np.unique(np.quantile(series, np.linspace(0, 1, bins + 1)))
    binned = pd.cut(series, bins=edges, include_lowest=True, duplicates="drop")
    table = _woe_iv_table(binned, target)
    return edges, table


def fit_woe_categorical(series: pd.Series, target: pd.Series):
    table = _woe_iv_table(series, target)
    return table


def apply_woe_numeric(series: pd.Series, edges, table: pd.DataFrame) -> pd.Series:
    binned = pd.cut(series, bins=edges, include_lowest=True, duplicates="drop")
    mapping = table["woe"].to_dict()
    return binned.map(mapping).astype(float).fillna(0.0)


def apply_woe_categorical(series: pd.Series, table: pd.DataFrame) -> pd.Series:
    mapping = table["woe"].to_dict()
    return series.map(mapping).astype(float).fillna(0.0)


NUMERIC_FEATURES = [
    "AGE", "AMT_INCOME_TOTAL", "AMT_CREDIT", "DURATION_MONTHS",
    "EMPLOYMENT_YEARS", "NUM_DEPENDENTS", "EXISTING_CREDITS",
    "DELINQUENCIES_2YR", "DEBT_TO_INCOME", "CREDIT_UTILIZATION",
]
CATEGORICAL_FEATURES = ["CHECKING_STATUS", "SAVINGS_STATUS", "PURPOSE", "HOUSING"]


def build_woe_features(df: pd.DataFrame, target_col: str = "TARGET"):
    """
    Fits WOE encoders on the given dataframe and returns:
      - woe_df: dataframe of WOE-transformed features (prefixed WOE_)
      - encoders: dict needed to transform new/unseen data the same way
      - iv_summary: dataframe ranking features by total IV
    """
    y = df[target_col]
    woe_df = pd.DataFrame(index=df.index)
    encoders = {"numeric": {}, "categorical": {}}
    iv_rows = []

    for col in NUMERIC_FEATURES:
        edges, table = fit_woe_numeric(df[col], y)
        woe_df[f"WOE_{col}"] = apply_woe_numeric(df[col], edges, table)
        encoders["numeric"][col] = (edges, table)
        iv_rows.append({"feature": col, "iv": table["iv"].sum()})

    for col in CATEGORICAL_FEATURES:
        table = fit_woe_categorical(df[col], y)
        woe_df[f"WOE_{col}"] = apply_woe_categorical(df[col], table)
        encoders["categorical"][col] = table
        iv_rows.append({"feature": col, "iv": table["iv"].sum()})

    iv_summary = pd.DataFrame(iv_rows).sort_values("iv", ascending=False).reset_index(drop=True)
    return woe_df, encoders, iv_summary


def transform_woe_features(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Apply already-fitted WOE encoders to new data (e.g. a holdout set)."""
    woe_df = pd.DataFrame(index=df.index)
    for col, (edges, table) in encoders["numeric"].items():
        woe_df[f"WOE_{col}"] = apply_woe_numeric(df[col], edges, table)
    for col, table in encoders["categorical"].items():
        woe_df[f"WOE_{col}"] = apply_woe_categorical(df[col], table)
    return woe_df
