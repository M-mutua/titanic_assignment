import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def remove_high_correlation(df: pd.DataFrame, threshold: float = 0.9):
    corr = df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    reduced = df.drop(columns=to_drop, errors="ignore")
    return reduced, to_drop


def random_forest_importance(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> pd.DataFrame:
    model = RandomForestClassifier(
        n_estimators=500,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X, y)
    importance_df = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return importance_df


def run_rfe(X: pd.DataFrame, y: pd.Series, n_features_to_select: int):
    estimator = LogisticRegression(max_iter=2000)
    selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
    selector.fit(X, y)
    selected = list(X.columns[selector.support_])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best Titanic features.")
    parser.add_argument("--input-dir", type=str, default="../data", help="Directory containing train_features.csv")
    parser.add_argument("--output-dir", type=str, default="../data", help="Directory for selected outputs")
    parser.add_argument("--corr-threshold", type=float, default=0.9, help="Correlation threshold for dropping features")
    parser.add_argument("--top-k", type=int, default=20, help="Top K features to keep from Random Forest ranking")
    parser.add_argument("--run-rfe", action="store_true", help="Run optional RFE and export its selection")
    args = parser.parse_args()

    input_dir = Path(__file__).resolve().parent / args.input_dir
    output_dir = Path(__file__).resolve().parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_features_path = input_dir / "train_features.csv"
    test_features_path = input_dir / "test_features.csv"

    if not train_features_path.exists():
        raise FileNotFoundError(f"Missing required file: {train_features_path}")

    train_df = pd.read_csv(train_features_path)
    test_df = pd.read_csv(test_features_path) if test_features_path.exists() else None

    if "Survived" not in train_df.columns:
        raise ValueError("train_features.csv must contain 'Survived' column")

    y = train_df["Survived"].astype(int)
    X = train_df.drop(columns=["Survived"])

    X_reduced, dropped_corr = remove_high_correlation(X, threshold=args.corr_threshold)
    importance_df = random_forest_importance(X_reduced, y)

    top_k = min(args.top_k, len(importance_df))
    selected_features = importance_df.head(top_k)["feature"].tolist()

    selected_train = train_df[["Survived"] + selected_features]
    selected_train_out = output_dir / "train_selected.csv"
    selected_train.to_csv(selected_train_out, index=False)

    selected_test_out = None
    if test_df is not None:
        missing_cols = [c for c in selected_features if c not in test_df.columns]
        for col in missing_cols:
            test_df[col] = 0
        selected_test = test_df[selected_features]
        selected_test_out = output_dir / "test_selected.csv"
        selected_test.to_csv(selected_test_out, index=False)

    importance_out = output_dir / "feature_importance.csv"
    importance_df.to_csv(importance_out, index=False)

    selected_txt_out = output_dir / "selected_features.txt"
    with selected_txt_out.open("w", encoding="utf-8") as f:
        f.write("Selected features (Random Forest top-ranked):\n")
        for feat in selected_features:
            f.write(f"- {feat}\n")

        f.write("\nDropped due to high correlation:\n")
        if dropped_corr:
            for feat in dropped_corr:
                f.write(f"- {feat}\n")
        else:
            f.write("- None\n")

    print(f"Saved: {selected_train_out}")
    if selected_test_out:
        print(f"Saved: {selected_test_out}")
    print(f"Saved: {importance_out}")
    print(f"Saved: {selected_txt_out}")

    if args.run_rfe:
        rfe_k = min(10, X_reduced.shape[1])
        rfe_selected = run_rfe(X_reduced, y, n_features_to_select=rfe_k)
        rfe_out = output_dir / "rfe_selected_features.txt"
        with rfe_out.open("w", encoding="utf-8") as f:
            f.write("RFE selected features:\n")
            for feat in rfe_selected:
                f.write(f"- {feat}\n")
        print(f"Saved: {rfe_out}")


if __name__ == "__main__":
    main()
