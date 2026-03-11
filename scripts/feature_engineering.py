"""
feature_engineering.py
-----------------------
Derives new features from the cleaned Titanic dataset.

All transforms are pure functions that take a DataFrame and return a new one.
Call build_features() to run the full pipeline.
"""

import numpy as np
import pandas as pd
import re


# ── Derived features ──────────────────────────────────────────────────────────

def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    FamilySize: total people travelling together (self + siblings/spouse + parents/children).
    IsAlone: 1 when travelling solo — research shows this correlates with lower survival.
    """
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def extract_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pull the title (Mr, Mrs, Miss, etc.) out of the Name field.
    Rare titles are grouped into 'Rare' to avoid sparse one-hot columns.
    """
    df = df.copy()

    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()

    rare_titles = {
        "Lady", "Countess", "Capt", "Col", "Don", "Dr",
        "Major", "Rev", "Sir", "Jonkheer", "Dona"
    }
    df["Title"] = df["Title"].replace(list(rare_titles), "Rare")
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    return df


def extract_deck(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deck is the first character of the Cabin value (A–G + T).
    Unknown cabin → 'Unknown'.
    """
    df = df.copy()
    df["Deck"] = df["Cabin"].str[0].fillna("Unknown")
    return df


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin Age into ordinal groups.
    Boundaries chosen to match common domain splits used in Titanic literature.
    """
    df = df.copy()
    bins = [0, 12, 17, 60, 80]
    labels = ["Child", "Teen", "Adult", "Senior"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True)
    return df


def add_fare_per_person(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fare in the dataset is often a shared ticket price.
    Dividing by FamilySize gives a per-person estimate.
    Assumes FamilySize has already been computed.
    """
    df = df.copy()
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
    return df


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode nominal columns.
    Pclass is already ordinal (1 > 2 > 3) so we leave it as integer.
    drop_first=True prevents the dummy variable trap.
    """
    df = df.copy()
    nominal_cols = ["Sex", "Embarked", "Title", "Deck", "AgeGroup"]
    existing = [c for c in nominal_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing, drop_first=True)
    return df


# ── Transformations ───────────────────────────────────────────────────────────

def log_transform_skewed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fare and FarePerPerson are right-skewed. Log1p (log(x+1)) handles zeros
    without blowing up and pulls the tail in.
    Age is less skewed after capping, but a log transform is still mildly helpful.
    """
    df = df.copy()
    for col in ["Fare", "FarePerPerson"]:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col])
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional interaction terms. These can help tree models but may add noise for
    linear models. Include only if feature selection keeps them.
    """
    df = df.copy()
    if "Fare" in df.columns:
        df["Pclass_x_Fare"] = df["Pclass"] * df["Fare"]
    return df


# ── Pipeline ──────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, encode: bool = True) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.
    Set encode=False to skip one-hot encoding (useful for inspection).
    """
    df = add_family_features(df)
    df = extract_title(df)
    df = extract_deck(df)
    df = add_age_group(df)
    df = add_fare_per_person(df)
    df = log_transform_skewed(df)
    df = add_interaction_features(df)
    if encode:
        df = encode_categorical(df)
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from data_cleaning import load_data, clean

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base, "data", "train.csv")
    out_path = os.path.join(base, "data", "train_engineered.csv")

    df = load_data(raw_path)
    df = clean(df)
    df = build_features(df)
    df.to_csv(out_path, index=False)
    print(f"[features] Saved to {out_path}")
    print(f"[features] New columns: {list(df.columns)}")
