"""
SHAP explainability for the LightGBM model.

Why this matters for credit scoring specifically: regulators (e.g. under
ECOA/Reg B in the US, or GDPR "right to explanation" in the EU) require
lenders to be able to state the specific reasons a credit decision was made.
SHAP gives per-applicant, per-feature attributions that map directly to
"adverse action reason codes."
"""

import pickle
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import shap

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def main():
    with open(OUT / "gbm_model.pkl", "rb") as f:
        d = pickle.load(f)
    gbm, features = d["model"], d["features"]

    df = pd.read_csv(ROOT / "data" / "application_data.csv")
    cat_cols = ["CHECKING_STATUS", "SAVINGS_STATUS", "PURPOSE", "HOUSING"]
    X = df[features].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")

    sample = X.sample(n=min(2000, len(X)), random_state=42)

    explainer = shap.TreeExplainer(gbm)
    shap_values = explainer.shap_values(sample)

    # Global importance
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False, plot_size=(8, 6))
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Per-applicant explanation (first row) -> "adverse action reason codes" style
    plt.figure()
    exp = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=sample.iloc[0],
        feature_names=sample.columns.tolist(),
    )
    shap.plots.waterfall(exp, show=False, max_display=10)
    plt.tight_layout()
    plt.savefig(OUT / "shap_waterfall_example.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved shap_summary.png and shap_waterfall_example.png to {OUT}/")


if __name__ == "__main__":
    main()
