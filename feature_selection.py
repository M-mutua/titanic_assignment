"""
feature_selection.py
---------------------
Selects the most informative features using:
  1. Correlation analysis (drop highly correlated pairs)
  2. Random Forest feature importance
  3. Optional: Recursive Feature Elimination (RFE)

Outputs a ranked feature list and a trimmed DataFrame ready for modelling.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder


# ── Correlation filter ────────────────────────────────────────────────────────

def drop_correlated_features(
    df: pd.DataFrame,
    target: str,
    threshold: float = 0.90,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove one feature from each pair whose Pearson correlation exceeds
    the threshold. We always keep the feature more correlated with the target.

    Returns the trimmed DataFrame and a list of dropped column names.
    """
    numeric = df.select_dtypes(include=[np.number]).drop(columns=[target], errors="ignore")
    corr = numeric.corr().abs()

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = []

    for col in upper.columns:
        pairs = upper[col][upper[col] > threshold].index.tolist()
        for paired in pairs:
            # Keep whichever is more correlated with the survival target
            if target in df.columns:
                corr_col = abs(df[col].corr(df[target]))
                corr_paired = abs(df[paired].corr(df[target]))
                drop_col = col if corr_col < corr_paired else paired
            else:
                drop_col = col
            if drop_col not in to_drop:
                to_drop.append(drop_col)

    df_trimmed = df.drop(columns=to_drop, errors="ignore")
    print(f"[select] Dropped {len(to_drop)} highly correlated feature(s): {to_drop}")
    return df_trimmed, to_drop


# ── Random Forest importance ──────────────────────────────────────────────────

def get_rf_importances(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 200,
    random_state: int = 42,
) -> pd.Series:
    """
    Fit a Random Forest and return feature importances as a sorted Series.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    return importances.sort_values(ascending=False)


def plot_importances(importances: pd.Series, top_n: int = 20, out_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    importances.head(top_n).sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)")
    ax.set_xlabel("Mean Decrease in Impurity")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"[select] Importance plot saved to {out_path}")
    plt.close(fig)


# ── RFE (optional) ────────────────────────────────────────────────────────────

def run_rfe(
    X: pd.DataFrame,
    y: pd.Series,
    n_features_to_select: int = 15,
    random_state: int = 42,
) -> list[str]:
    """
    Run Recursive Feature Elimination with a Random Forest estimator.
    Returns the list of selected feature names.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rfe = RFE(estimator=rf, n_features_to_select=n_features_to_select, step=1)
    rfe.fit(X, y)
    selected = X.columns[rfe.support_].tolist()
    print(f"[select] RFE selected {len(selected)} features: {selected}")
    return selected


# ── Prep helpers ──────────────────────────────────────────────────────────────

def get_model_ready(df: pd.DataFrame, target: str = "Survived") -> tuple[pd.DataFrame, pd.Series]:
    """
    Drop non-numeric / ID columns that have no predictive value,
    and split X / y.
    """
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin", target]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Encode any remaining object columns (shouldn't be any after feature_engineering)
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Encode category dtype (e.g. AgeGroup if encode=False was used)
    for col in X.select_dtypes(include="category").columns:
        X[col] = X[col].cat.codes

    X = X.fillna(0)
    y = df[target] if target in df.columns else None
    return X, y


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_cleaning import load_data, clean
    from feature_engineering import build_features

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base, "data", "train.csv")

    df = load_data(raw_path)
    df = clean(df)
    df = build_features(df)

    df, dropped = drop_correlated_features(df, target="Survived")
    X, y = get_model_ready(df)

    importances = get_rf_importances(X, y)
    print("\nTop 20 features by RF importance:")
    print(importances.head(20))

    plot_path = os.path.join(base, "data", "feature_importances.png")
    plot_importances(importances, out_path=plot_path)

    # Save final selected features list
    top_features = importances[importances > 0.01].index.tolist()
    print(f"\n[select] Features with importance > 1%: {top_features}")

    # Optional RFE
    rfe_features = run_rfe(X, y, n_features_to_select=15)
