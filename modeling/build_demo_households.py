"""Build models/demo_households.json: everything the web page shows, taken from the dataset.

Run from the project root (needs data/raw/... and data/processed/model_state.csv, see notebook 02):

    python modeling/build_demo_households.py

Nothing here is typed by hand. Where each part comes from:
  labels      base_categories: Category_ID -> Category name
  options     the values each dropdown can take = the distinct values found in model_state.csv for that column
  households  four real household states, one per preset button, taken ONLY from the held-out households
              (validation + test of the grouped split in splits.py), so the Random Forests never saw them in training.
              The original page's presets described households with 1, 7 and 21 documented items, so for each of
              those sizes we take the real held-out state whose documented_asset_count is closest
              (ties: lowest Property_ID, then earliest Prediction_Time).
              A fourth button, "Large inventory", is one chosen household: PRP-000675 (a test household), in its
              latest state, which has 80 documented items. It is the only button whose household was named by hand.
              features = that state's row exactly as the recommender was trained on it (blank stays blank)
              items    = the assets that state counts: assets.Created_At <= Prediction_Time for that property,
                         with the real Asset_Name, Category and Current_Estimated_Value
The script checks that the item list reproduces the state's counts and documented value before it writes.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modeling"))

from splits import split_properties  # noqa: E402
WORKBOOK = ROOT / "data" / "raw" / "myVal_Synthetic_Datasets_release_v2.xlsx"
MODEL_STATE = ROOT / "data" / "processed" / "model_state.csv"
OUTPUT = ROOT / "models" / "demo_households.json"

NOT_FEATURES = ["Property_ID", "Customer_ID", "Prediction_Time", "Next_Category_ID", "Next_Category"]
PRESET_SIZES = {"Just started": 1, "Typical household": 7, "Well documented": 21}  # item counts of the original page's presets
NAMED_HOUSEHOLDS = {"Large inventory": "PRP-000675"}  # chosen by hand; its latest state (the one with the most items) is used
CATEGORICAL = [
    "Property_Type", "Occupancy_Type", "Insurance_Coverage_Type", "Age_Band", "Gender", "State", "Occupation",
    "Household_Type", "Income_Band", "Financial_Literacy_Band", "Engagement_Tier", "Membership_Level",
    "Account_Status", "Preferred_Channel", "Preferred_Explanation_Style", "Prior_Claim_Experience",
    "Underinsurance_Awareness",
]


def plain(value):
    """numpy / pandas value -> JSON value (blank stays None)."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NaT:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def main():
    state = pd.read_csv(MODEL_STATE, parse_dates=["Prediction_Time"])
    sheets = pd.read_excel(WORKBOOK, sheet_name=["assets", "base_categories"])
    assets = sheets["assets"]
    assets["Created_At"] = pd.to_datetime(assets["Created_At"], errors="coerce")
    categories = sheets["base_categories"]

    labels = dict(zip(categories["Category_ID"], categories["Name"]))
    options = {c: sorted(state[c].dropna().astype(str).unique()) for c in CATEGORICAL}

    split = split_properties(state)
    held_out = split["validation"] | split["test"]  # never used to train the recommender
    candidates = state[state["Property_ID"].isin(held_out)]

    rows = {}
    for name, target in PRESET_SIZES.items():
        ranked = candidates.assign(gap=(state["documented_asset_count"] - target).abs())
        rows[name] = ranked.sort_values(["gap", "Property_ID", "Prediction_Time"]).iloc[0]
    for name, property_id in NAMED_HOUSEHOLDS.items():
        assert property_id in held_out, (name, property_id, "is not a validation or test household")
        own = candidates[candidates["Property_ID"] == property_id]
        rows[name] = own.sort_values(["documented_asset_count", "Prediction_Time"]).iloc[-1]

    households = {}
    for name, row in rows.items():
        owned = assets[(assets["Property_ID"] == row["Property_ID"]) & (assets["Created_At"] <= row["Prediction_Time"])]
        owned = owned.sort_values(["Created_At", "Asset_ID"])
        assert len(owned) == row["documented_asset_count"], (name, "item count does not match the state")
        assert abs(owned["Current_Estimated_Value"].sum() - row["documented_value_total"]) < 1e-6, (name, "value total does not match")
        households[name] = {
            "property_id": row["Property_ID"],
            "prediction_time": str(row["Prediction_Time"]),
            "features": {c: plain(row[c]) for c in state.columns if c not in NOT_FEATURES},
            "items": [
                {
                    "asset_id": r.Asset_ID,
                    "name": r.Asset_Name,
                    "category": r.Category,
                    "category_id": r.Category_ID,
                    "value": plain(r.Current_Estimated_Value),
                }
                for r in owned.itertuples()
            ],
        }
        print(f"{name}: {row['Property_ID']} at {row['Prediction_Time']}, {len(owned)} items, "
              f"documented value A${row['documented_value_total']:,.0f}")

    # How often the true next category was in the ranking model's top 2 and top 3 on the test households,
    # so the page does not hard-code the figure it quotes.
    from my_val_capstone_01.modeling.predict import ranking_model

    test_rows = state[state["Property_ID"].isin(split["test"])]
    probabilities = ranking_model.predict_proba(test_rows.drop(columns=NOT_FEATURES))
    classes = ranking_model.named_steps["classifier"].classes_
    order = np.argsort(-probabilities, axis=1)
    truth = test_rows["Next_Category"].to_numpy()
    holdout = {"states": int(len(test_rows))}
    for k in (2, 3):
        holdout[f"top_{k}"] = round(float(np.mean([truth[i] in classes[order[i, :k]] for i in range(len(truth))])), 4)
    print(f"holdout (test households): {holdout}")

    with open(OUTPUT, "w") as handle:
        json.dump({"labels": labels, "options": options, "households": households, "holdout": holdout,
                   "built_from": "model_state.csv, assets and base_categories sheets"}, handle, indent=1)
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
