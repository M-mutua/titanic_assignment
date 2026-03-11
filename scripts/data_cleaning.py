"""
data_cleaning.py
----------------
Handles all data cleaning steps for the Titanic dataset:
  - Missing value imputation
  - Outlier capping
  - Data type / consistency fixes
  - Duplicate removal

Outputs: data/train_cleaned.csv
"""

import pandas as pd
import numpy as np


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy summary of missing values per column."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    return pd.DataFrame({"missing_count": missing, "missing_pct": pct}).query(
        "missing_count > 0"
    )


# ── Cleaning steps ───────────────────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy per column:
      Age      → median imputation + binary flag HasAge
      Embarked → mode imputation (only 2 missing, both port known from context)
      Fare     → median imputation (1 missing in test set)
      Cabin    → kept as-is; Deck extraction in feature_engineering will handle it
    """
    df = df.copy()

    # Age: median is more robust than mean for right-skewed data
    df["HasAge"] = df["Age"].notna().astype(int)
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Embarked: 2 rows, mode is 'S'
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Fare: mostly affects test.csv but guard against it here too
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Cabin: too sparse (~77% missing) to impute meaningfully.
    # We extract the deck letter in feature_engineering; leave NaN for now.

    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap extreme values rather than drop rows — we can't afford to lose
    survival labels in a small dataset.

    Fare: anything above the 99th percentile is capped.
    Age:  biologically bounded (0-80 is a reasonable hard cap).
    """
    df = df.copy()

    fare_cap = df["Fare"].quantile(0.99)
    df["Fare"] = df["Fare"].clip(upper=fare_cap)

    df["Age"] = df["Age"].clip(lower=0, upper=80)

    return df


def fix_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise string columns to lowercase and strip whitespace.
    Sex should only have 'male' / 'female'.
    """
    df = df.copy()

    df["Sex"] = df["Sex"].str.strip().str.lower()
    df["Embarked"] = df["Embarked"].str.strip().str.upper()

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        print(f"[clean] Removed {dropped} duplicate row(s).")
    return df


# ── Pipeline ─────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_duplicates(df)
    df = fix_consistency(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base, "data", "train.csv")
    out_path = os.path.join(base, "data", "train_cleaned.csv")

    df_raw = load_data(raw_path)
    print("Missing values before cleaning:")
    print(report_missing(df_raw))

    df_clean = clean(df_raw)
    df_clean.to_csv(out_path, index=False)
    print(f"\n[clean] Saved cleaned dataset to {out_path}")
    print(f"[clean] Shape: {df_clean.shape}")
