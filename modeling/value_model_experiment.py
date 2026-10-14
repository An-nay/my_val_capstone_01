"""Can a model predict the value of the next item better than the category median?

Run from the project root (needs data/raw/... and data/processed/model_state.csv, see notebook 02):

    python modeling/value_model_experiment.py

Read-only: it trains throwaway models in memory, writes no files and does not touch the saved models.
All scores use grouped 5-fold cross-validation by Property_ID on the TRAINING and VALIDATION households
only (the split of 03_fm_modelling); the test households are never read.

Experiment 1, "next item": the target is the value of the item the customer actually documented next
(Current_Estimated_Value of the target category's first asset event, median if several), for the same
prediction states as the classifier. Inputs are the classifier's own features plus the next category.

Experiment 2, "all items": the same question on every valued, non-vehicle asset, with the category plus
the household's static details (property, customer and profile fields), which gives far more rows.

In both, the target is log(value) and the baseline is the training-fold median per category (what the
suggestion loop uses). Scores are in dollars except "log error", the mean absolute error on log(value).
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modeling"))

from splits import split_properties  # noqa: E402

WORKBOOK = ROOT / "data" / "raw" / "myVal_Synthetic_Datasets_release_v2.xlsx"
MODEL_STATE = ROOT / "data" / "processed" / "model_state.csv"
BASELINE = "category median (today)"
ID_COLUMNS = ["Property_ID", "Customer_ID", "Prediction_Time", "Next_Category_ID", "Next_Category"]


def _pipeline(model, num, cat):
    numeric = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())])
    categorical = Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("e", OneHotEncoder(handle_unknown="ignore"))])
    return Pipeline([("p", ColumnTransformer([("n", numeric, num), ("c", categorical, cat)])), ("m", model)])


def _split_columns(frame, columns):
    num = [c for c in columns if is_numeric_dtype(frame[c]) and not is_bool_dtype(frame[c])]
    return num, [c for c in columns if c not in num]


def _cross_validate(X, y, groups, category, models):
    """Out-of-fold predictions (in log space) for the baseline and every model."""
    out = {name: np.zeros(len(y)) for name in [BASELINE, *models]}
    for train, test in GroupKFold(n_splits=5).split(X, y, groups):
        medians = pd.Series(y[train]).groupby(category[train]).median()
        out[BASELINE][test] = pd.Series(category[test]).map(medians).fillna(np.median(y[train])).values
        for name, model in models.items():
            model.fit(X.iloc[train], y[train])
            out[name][test] = model.predict(X.iloc[test])
    return out


def _report(title, y, predictions, mask=None):
    mask = np.ones(len(y), bool) if mask is None else mask
    truth = np.exp(y[mask])
    print(f"\n{title} (n={mask.sum()})")
    print(f"{'method':32s} {'mean err $':>10s} {'median err $':>13s} {'log error':>10s} {'within 50%':>11s} {'within 25%':>11s}")
    for name, log_pred in predictions.items():
        pred = np.exp(log_pred[mask])
        error = np.abs(pred - truth)
        print(
            f"{name:32s} {error.mean():10.0f} {np.median(error):13.0f} {np.abs(log_pred[mask] - y[mask]).mean():10.3f} "
            f"{np.mean((pred >= 0.5 * truth) & (pred <= 1.5 * truth)):11.1%} {np.mean((pred >= 0.75 * truth) & (pred <= 1.25 * truth)):11.1%}"
        )


def next_item_experiment(sheets, state, parts):
    assets = sheets["assets"]
    assets["Created_At"] = pd.to_datetime(assets["Created_At"], errors="coerce")
    first = assets.groupby(["Property_ID", "Category"])["Created_At"].min().rename("t").reset_index()
    events = assets.merge(first, on=["Property_ID", "Category"])
    events = events[events["Created_At"] == events["t"]]
    value = events.groupby(["Property_ID", "Category"])["Current_Estimated_Value"].median().rename("next_value").reset_index()
    data = state.merge(value, left_on=["Property_ID", "Next_Category"], right_on=["Property_ID", "Category"], how="left").drop(columns="Category")
    data = data[data["next_value"].notna() & (data["next_value"] > 0)]
    dev = data[data["Property_ID"].isin(parts["train"] | parts["validation"])].reset_index(drop=True)

    columns = [c for c in state.columns if c not in ID_COLUMNS] + ["Next_Category"]
    X, y, groups = dev[columns].copy(), np.log(dev["next_value"].values), dev["Property_ID"].values
    num, cat = _split_columns(X, columns)
    for c in cat:
        X[c] = X[c].astype(str)
    models = {
        "Ridge": _pipeline(Ridge(alpha=10), num, cat),
        "RandomForest": _pipeline(RandomForestRegressor(n_estimators=400, min_samples_leaf=5, random_state=42, n_jobs=-1), num, cat),
        "GradientBoosting": _pipeline(GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42), num, cat),
    }
    predictions = _cross_validate(X, y, groups, dev["Next_Category"].values, models)
    print("EXPERIMENT 1: value of the item actually documented next (same states as the classifier)")
    _report("all next items", y, predictions)
    _report("excluding Vehicles (never suggested)", y, predictions, dev["Next_Category"].values != "Vehicles")


def all_items_experiment(sheets, state, parts):
    assets, properties, customers, profiles = (sheets[k] for k in ("assets", "properties", "customers", "customer_profiles"))
    property_cols = ["Property_Type", "Occupancy_Type", "Number_Of_Occupants", "Number_Of_Bedrooms", "Property_Valuation_AUD", "Is_Home_Insured", "Insurance_Coverage_Type"]
    data = assets[["Asset_ID", "Property_ID", "Customer_ID", "Category", "Current_Estimated_Value"]]
    data = data.merge(properties[["Property_ID", *property_cols]], on="Property_ID")
    data = data.merge(customers[["Customer_ID", "Age_Band", "Gender", "State", "Membership_Level"]], on="Customer_ID")
    data = data.merge(profiles[["Customer_ID", "Financial_Literacy_Band", "Household_Type", "Income_Band"]], on="Customer_ID")
    data = data[data["Current_Estimated_Value"].notna() & (data["Current_Estimated_Value"] > 0) & (data["Category"] != "Vehicles")]
    dev = data[data["Property_ID"].isin(parts["train"] | parts["validation"])].reset_index(drop=True)

    columns = ["Category", *property_cols, "Age_Band", "Gender", "State", "Membership_Level", "Financial_Literacy_Band", "Household_Type", "Income_Band"]
    X, y, groups = dev[columns].copy(), np.log(dev["Current_Estimated_Value"].values), dev["Property_ID"].values
    num, cat = _split_columns(X, columns)
    for c in cat:
        X[c] = X[c].astype(str)
    models = {
        "Ridge": _pipeline(Ridge(alpha=30), num, cat),
        "RandomForest": _pipeline(RandomForestRegressor(n_estimators=300, min_samples_leaf=20, random_state=42, n_jobs=-1), num, cat),
        "LightGBM": _pipeline(LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=8, min_child_samples=40, subsample=0.8, colsample_bytree=0.8, verbose=-1, random_state=42), num, cat),
    }
    predictions = _cross_validate(X, y, groups, dev["Category"].values, models)
    print("\n\nEXPERIMENT 2: value of any item, from category and household details")
    _report(f"all valued non-vehicle assets, {dev['Property_ID'].nunique()} households", y, predictions)
    print("\nspread of real item values inside each category (25th / median / 75th percentile):")
    print(dev.groupby("Category")["Current_Estimated_Value"].quantile([0.25, 0.5, 0.75]).unstack().round(0).to_string())


def main():
    sheets = pd.read_excel(WORKBOOK, sheet_name=["assets", "properties", "customers", "customer_profiles"])
    state = pd.read_csv(MODEL_STATE, parse_dates=["Prediction_Time"])
    parts = split_properties(state)
    next_item_experiment(sheets, state, parts)
    all_items_experiment(sheets, state, parts)


if __name__ == "__main__":
    main()
