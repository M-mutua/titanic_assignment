import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


COMMON_TITLES = {"Mr", "Mrs", "Miss", "Master"}


def extract_title(name: str) -> str:
    if pd.isna(name):
        return "Unknown"
    if "," in name and "." in name:
        return name.split(",", 1)[1].split(".", 1)[0].strip()
    return "Unknown"


def normalize_title(raw_title: str) -> str:
    if raw_title in COMMON_TITLES:
        return raw_title
    if raw_title in ["Mlle", "Ms"]:
        return "Miss"
    if raw_title in ["Mme"]:
        return "Mrs"
    return "Rare"


def extract_deck(cabin: str) -> str:
    if pd.isna(cabin):
        return "Unknown"
    cabin = str(cabin).strip()
    if not cabin or cabin.lower() == "unknown":
        return "Unknown"
    return cabin[0].upper()


def age_group(age: float) -> str:
    if age < 13:
        return "Child"
    if age < 20:
        return "Teen"
    if age < 60:
        return "Adult"
    return "Senior"


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

    out["Title"] = out["Name"].apply(extract_title).apply(normalize_title)
    out["Deck"] = out["Cabin"].apply(extract_deck)
    out["AgeGroup"] = out["Age"].apply(age_group)

    safe_family_size = out["FamilySize"].replace(0, 1)
    out["FarePerPerson"] = out["Fare"] / safe_family_size

    # Interaction examples.
    out["Pclass_Fare"] = out["Pclass"] * out["Fare"]
    out["Age_Pclass"] = out["Age"] * out["Pclass"]

    # Stabilize skewed numeric features.
    out["Fare_log"] = np.log1p(out["Fare"].clip(lower=0))
    out["Age_log"] = np.log1p(out["Age"].clip(lower=0))

    return out


def one_hot_and_scale(train_df: pd.DataFrame, test_df: pd.DataFrame | None = None):
    target_col = "Survived"

    train_y = train_df[target_col] if target_col in train_df.columns else None

    # Drop high-cardinality identifiers that are not useful as direct model inputs.
    cols_to_drop = ["Name", "Ticket", "Cabin", "TitleRaw"]
    train_x = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns], errors="ignore")
    test_x = None
    if test_df is not None:
        test_x = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], errors="ignore")

    categorical_cols = [
        c
        for c in ["Sex", "Embarked", "Title", "Deck", "AgeGroup"]
        if c in train_x.columns
    ]

    train_x = pd.get_dummies(train_x, columns=categorical_cols, drop_first=False)
    if test_x is not None:
        test_x = pd.get_dummies(test_x, columns=categorical_cols, drop_first=False)
        train_x, test_x = train_x.align(test_x, join="left", axis=1, fill_value=0)

    # Re-attach target for train.
    if train_y is not None:
        train_x[target_col] = train_y.values

    # Scale numeric columns excluding target and PassengerId.
    scale_exclude = {target_col, "PassengerId"}
    numeric_cols = [
        c for c in train_x.columns if c not in scale_exclude and np.issubdtype(train_x[c].dtype, np.number)
    ]

    scaler = StandardScaler()
    train_x[numeric_cols] = scaler.fit_transform(train_x[numeric_cols])

    if test_x is not None:
        test_numeric = [c for c in numeric_cols if c in test_x.columns]
        test_x[test_numeric] = scaler.transform(test_x[test_numeric])

    return train_x, test_x


def main() -> None:
    parser = argparse.ArgumentParser(description="Engineer Titanic features from cleaned data.")
    parser.add_argument("--input-dir", type=str, default="../data", help="Directory containing cleaned CSV files")
    parser.add_argument("--output-dir", type=str, default="../data", help="Directory for feature CSV files")
    args = parser.parse_args()

    input_dir = Path(__file__).resolve().parent / args.input_dir
    output_dir = Path(__file__).resolve().parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cleaned_path = input_dir / "train_cleaned.csv"
    test_cleaned_path = input_dir / "test_cleaned.csv"

    if not train_cleaned_path.exists():
        raise FileNotFoundError(f"Missing required file: {train_cleaned_path}")

    train_df = pd.read_csv(train_cleaned_path)
    test_df = pd.read_csv(test_cleaned_path) if test_cleaned_path.exists() else None

    train_eng = engineer(train_df)
    test_eng = engineer(test_df) if test_df is not None else None

    train_features, test_features = one_hot_and_scale(train_eng, test_eng)

    train_out = output_dir / "train_features.csv"
    train_features.to_csv(train_out, index=False)
    print(f"Saved: {train_out}")

    if test_features is not None:
        test_out = output_dir / "test_features.csv"
        test_features.to_csv(test_out, index=False)
        print(f"Saved: {test_out}")
    else:
        print("test_cleaned.csv not found. Skipping test feature export.")


if __name__ == "__main__":
    main()
