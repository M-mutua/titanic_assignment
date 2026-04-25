import argparse
from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_OUTLIER_COLS = ["Age", "Fare"]


def extract_title(name: str) -> str:
    if pd.isna(name):
        return "Unknown"
    if "," in name and "." in name:
        return name.split(",", 1)[1].split(".", 1)[0].strip()
    return "Unknown"


def cap_outliers_iqr(series: pd.Series, whisker: float = 1.5) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return series

    lower = q1 - whisker * iqr
    upper = q3 + whisker * iqr
    return series.clip(lower=lower, upper=upper)


def clean_dataframe(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    cleaned = df.copy()

    # Preserve missingness signal before imputation.
    cleaned["AgeWasMissing"] = cleaned["Age"].isna().astype(int)
    cleaned["FareWasMissing"] = cleaned["Fare"].isna().astype(int)
    cleaned["CabinWasMissing"] = cleaned["Cabin"].isna().astype(int)
    cleaned["EmbarkedWasMissing"] = cleaned["Embarked"].isna().astype(int)

    # Data consistency fixes.
    if "Sex" in cleaned.columns:
        cleaned["Sex"] = (
            cleaned["Sex"].astype(str).str.strip().str.lower().replace({"f": "female", "m": "male"})
        )
        cleaned.loc[~cleaned["Sex"].isin(["male", "female"]), "Sex"] = "unknown"

    if "Embarked" in cleaned.columns:
        cleaned["Embarked"] = cleaned["Embarked"].astype(str).str.strip()
        cleaned["Embarked"] = cleaned["Embarked"].replace({"": np.nan, "nan": np.nan})
        mode_embarked = cleaned["Embarked"].mode(dropna=True)
        if not mode_embarked.empty:
            cleaned["Embarked"] = cleaned["Embarked"].fillna(mode_embarked.iloc[0])

    # Impute fare with median by Pclass when possible.
    if "Fare" in cleaned.columns:
        fare_group_median = cleaned.groupby("Pclass")["Fare"].transform("median")
        global_fare_median = cleaned["Fare"].median()
        cleaned["Fare"] = cleaned["Fare"].fillna(fare_group_median).fillna(global_fare_median)

    # Impute age with median by Sex and Pclass (fallback global median).
    if "Age" in cleaned.columns:
        age_group_median = cleaned.groupby(["Sex", "Pclass"])["Age"].transform("median")
        global_age_median = cleaned["Age"].median()
        cleaned["Age"] = cleaned["Age"].fillna(age_group_median).fillna(global_age_median)

    # Keep Cabin for later deck extraction; replace missing with marker.
    if "Cabin" in cleaned.columns:
        cleaned["Cabin"] = cleaned["Cabin"].fillna("Unknown")

    # Optional helper for richer imputations later.
    cleaned["TitleRaw"] = cleaned["Name"].apply(extract_title)

    # Remove exact duplicates.
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    after = len(cleaned)
    if is_train and before != after:
        print(f"Removed duplicates: {before - after}")

    # Outlier capping.
    for col in NUMERIC_OUTLIER_COLS:
        if col in cleaned.columns:
            cleaned[col] = cap_outliers_iqr(cleaned[col])

    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Titanic train/test data.")
    parser.add_argument("--data-dir", type=str, default="../data", help="Directory containing train.csv and test.csv")
    parser.add_argument("--output-dir", type=str, default="../data", help="Directory for cleaned outputs")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent / args.data_dir
    output_dir = Path(__file__).resolve().parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing required file: {train_path}")

    train_df = pd.read_csv(train_path)
    print("Train missing values before cleaning:")
    print(train_df.isna().sum().sort_values(ascending=False))

    train_cleaned = clean_dataframe(train_df, is_train=True)
    train_out = output_dir / "train_cleaned.csv"
    train_cleaned.to_csv(train_out, index=False)
    print(f"Saved: {train_out}")

    if test_path.exists():
        test_df = pd.read_csv(test_path)
        print("\nTest missing values before cleaning:")
        print(test_df.isna().sum().sort_values(ascending=False))

        test_cleaned = clean_dataframe(test_df, is_train=False)
        test_out = output_dir / "test_cleaned.csv"
        test_cleaned.to_csv(test_out, index=False)
        print(f"Saved: {test_out}")
    else:
        print("test.csv not found. Skipping test cleaning.")


if __name__ == "__main__":
    main()
