# Titanic Survival Prediction — AI Assignment 2

Predictive modelling pipeline for the [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic), covering data cleaning, feature engineering, and feature selection.

---

## Project Structure

```
titanic_assignment/
├── data/
│   ├── train.csv               # Raw training data (download from Kaggle)
│   ├── test.csv                # Raw test data (optional)
│   └── train_cleaned.csv       # Output of data cleaning step
├── notebooks/
│   └── Titanic_Feature_Engineering.ipynb
├── scripts/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── feature_selection.py
├── README.md
└── requirements.txt
```

---

## Getting Started

### 1. Get the data

Download `train.csv` and `test.csv` from [Kaggle](https://www.kaggle.com/c/titanic/data) and place them in the `data/` folder.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

```bash
cd notebooks
jupyter notebook Titanic_Feature_Engineering.ipynb
```

### 4. Or run scripts directly

```bash
cd scripts

python3 data_cleaning.py        # Outputs data/train_cleaned.csv
python3 feature_engineering.py  # Outputs data/train_engineered.csv
python3 feature_selection.py    # Prints ranked features + saves importance plot
```

---

## Approach

### Data Cleaning

- **Age (~20% missing):** Filled with the median (more robust than mean for right-skewed data). Added a `HasAge` binary flag to preserve the information that age was missing.
- **Embarked (<1% missing):** Filled with mode ('S').
- **Cabin (~77% missing):** Not imputed — too sparse. The deck letter is extracted as a categorical feature in the engineering step.
- **Fare outliers:** Capped at the 99th percentile. One passenger paid £512 — capping prevents this from distorting scale-sensitive models.
- **Age outliers:** Capped at [0, 80] — biologically reasonable bounds.

### Features Engineered

| Feature | Description |
|---------|-------------|
| `FamilySize` | SibSp + Parch + 1 |
| `IsAlone` | 1 if travelling solo |
| `Title` | Extracted from Name (Mr, Mrs, Miss, Master, Rare) |
| `Deck` | First character of Cabin, Unknown if missing |
| `AgeGroup` | Binned: Child / Teen / Adult / Senior |
| `FarePerPerson` | Fare / FamilySize |
| `Fare_log` | log1p(Fare) — reduces right skew |
| `FarePerPerson_log` | log1p(FarePerPerson) |
| `Pclass_x_Fare` | Interaction term (optional) |

### Feature Selection

1. **Correlation filter:** Drop one feature from pairs with Pearson |r| > 0.90, keeping the feature more correlated with the target.
2. **Random Forest importance:** Rank all features by mean decrease in impurity across 200 trees.
3. **RFE (optional):** Recursive Feature Elimination with RF estimator, selecting the top 15 features.

### Key Findings

- `Sex` is the single strongest predictor — women survived at ~74% vs ~19% for men.
- `Title` compresses gender and social class into a single ordinal-like feature and ranks near the top.
- `Pclass` and `Fare_log` together capture socioeconomic status more cleanly than either alone.
- Solo travellers (`IsAlone = 1`) have significantly lower survival rates.
- `Cabin` is too sparse to use directly; `Deck` extracts the useful signal at the cost of a large `Unknown` category.
- `SibSp` and `Parch` individually are weaker than their composite `FamilySize` and are dropped.

---

## Requirements

See `requirements.txt`.
