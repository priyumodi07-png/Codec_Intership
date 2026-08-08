"""
End-to-end training pipeline:
  1. Load data
  2. WOE feature engineering
  3. Train baseline (Logistic Regression on WOE features) + LightGBM
  4. Evaluate with AUC, KS statistic, Gini coefficient
  5. Convert LR log-odds to a 300-850 scorecard
  6. Save everything needed for the Streamlit demo + SHAP notebook
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import lightgbm as lgb

from features import build_woe_features, transform_woe_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from scorecard import ScorecardScaler, risk_band

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def load_data():
    """Swap this to read the real Kaggle CSV -- see generate_data.py docstring."""
    path = ROOT / "data" / "application_data.csv"
    return pd.read_csv(path)


def ks_statistic(y_true, y_prob):
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False)
    df["cum_bad"] = (df["y"] == 1).cumsum() / (df["y"] == 1).sum()
    df["cum_good"] = (df["y"] == 0).cumsum() / (df["y"] == 0).sum()
    return float(np.max(np.abs(df["cum_bad"] - df["cum_good"])))


def gini_from_auc(auc):
    return 2 * auc - 1


def main():
    df = load_data()
    y = df["TARGET"]

    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=y)

    # ---- WOE features (fit on train only, applied to test) ----
    woe_train, encoders, iv_summary = build_woe_features(train_df)
    woe_test = transform_woe_features(test_df, encoders)

    iv_summary.to_csv(OUT / "iv_summary.csv", index=False)
    print("\n=== Information Value ranking ===")
    print(iv_summary.to_string(index=False))

    y_train, y_test = train_df["TARGET"].values, test_df["TARGET"].values

    # ---- Baseline: Logistic Regression on WOE features ----
    lr = LogisticRegression(max_iter=1000)
    lr.fit(woe_train, y_train)
    lr_train_prob = lr.predict_proba(woe_train)[:, 1]
    lr_test_prob = lr.predict_proba(woe_test)[:, 1]

    # ---- Advanced: LightGBM on raw features ----
    raw_features = NUMERIC_FEATURES.copy()
    cat_train = train_df[CATEGORICAL_FEATURES].astype("category")
    cat_test = test_df[CATEGORICAL_FEATURES].astype("category")
    X_train = pd.concat([train_df[raw_features], cat_train], axis=1)
    X_test = pd.concat([test_df[raw_features], cat_test], axis=1)

    gbm = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=15,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1
    )
    gbm.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)
    gbm_test_prob = gbm.predict_proba(X_test)[:, 1]

    # ---- Evaluation ----
    results = {}
    for name, prob in [("Logistic Regression (WOE)", lr_test_prob),
                        ("LightGBM", gbm_test_prob)]:
        auc = roc_auc_score(y_test, prob)
        ks = ks_statistic(y_test, prob)
        gini = gini_from_auc(auc)
        results[name] = {"AUC": round(auc, 4), "KS": round(ks, 4), "Gini": round(gini, 4)}

    print("\n=== Model comparison (holdout test set) ===")
    print(pd.DataFrame(results).T.to_string())
    with open(OUT / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- ROC curve plot ----
    plt.figure(figsize=(6, 6))
    for name, prob in [("Logistic Regression (WOE)", lr_test_prob), ("LightGBM", gbm_test_prob)]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "roc_curve.png", dpi=150)
    plt.close()

    # ---- Scorecard conversion (from LR model) ----
    scaler = ScorecardScaler(base_score=680, base_odds=20, pdo=60)
    test_scores = scaler.prob_to_score(lr_test_prob)
    score_df = test_df[["SK_ID_CURR", "TARGET"]].copy()
    score_df["PROB_DEFAULT"] = lr_test_prob
    score_df["SCORE"] = test_scores
    score_df["RISK_BAND"] = score_df["SCORE"].apply(risk_band)
    score_df.to_csv(OUT / "scored_test_set.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(score_df.loc[score_df.TARGET == 0, "SCORE"], bins=30, alpha=0.6, label="Good (no default)")
    plt.hist(score_df.loc[score_df.TARGET == 1, "SCORE"], bins=30, alpha=0.6, label="Bad (defaulted)")
    plt.xlabel("Credit Score")
    plt.ylabel("Count")
    plt.title("Score Distribution by Actual Outcome")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "score_distribution.png", dpi=150)
    plt.close()

    print("\n=== Score distribution by risk band ===")
    print(score_df.groupby("RISK_BAND")["TARGET"].agg(["count", "mean"])
          .rename(columns={"mean": "default_rate"}).sort_values("default_rate"))

    # ---- Persist models + encoders for the Streamlit app / SHAP notebook ----
    with open(OUT / "lr_model.pkl", "wb") as f:
        pickle.dump({"model": lr, "encoders": encoders, "scaler": scaler}, f)
    with open(OUT / "gbm_model.pkl", "wb") as f:
        pickle.dump({"model": gbm, "features": raw_features + CATEGORICAL_FEATURES}, f)

    print(f"\nArtifacts saved to {OUT}/")


if __name__ == "__main__":
    main()
