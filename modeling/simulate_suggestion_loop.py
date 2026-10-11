"""Replay the suggestion loop on real held-out households and report how long it keeps going.

Run from the project root (needs data/raw/..., data/processed/model_state.csv and the models/ folder):

    python modeling/simulate_suggestion_loop.py

For each validation and test household we start from its earliest real state and play the loop.
The "customer" says yes when the suggestion matches what the household really owns, which we know
from its full asset history:
  - a new category is accepted if the household ever documents an item in it
  - "more of a category" is accepted if the household ends up with more items there than it has now
A yes adds an item with the suggested value, the features are updated and the recommender runs again.
The first no ends the loop, as does running out of suggestions. Nothing here trains a model.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modeling.splits import split_properties  # noqa: E402
from my_val_capstone_01.modeling.notify import suggest_next_item  # noqa: E402
from my_val_capstone_01.modeling.predict import predict_next_categories  # noqa: E402
from my_val_capstone_01.modeling.state_features import apply_logged_item, counts_by_name  # noqa: E402

WORKBOOK = ROOT / "data" / "raw" / "myVal_Synthetic_Datasets_release_v2.xlsx"
MODEL_STATE = ROOT / "data" / "processed" / "model_state.csv"
NOT_FEATURES = ["Property_ID", "Customer_ID", "Prediction_Time", "Next_Category_ID", "Next_Category"]
MAX_STEPS = 60


def main():
    assets = pd.read_excel(WORKBOOK, sheet_name="assets")
    assets["Created_At"] = pd.to_datetime(assets["Created_At"], errors="coerce")
    state = pd.read_csv(MODEL_STATE, parse_dates=["Prediction_Time"])
    held_out = split_properties(state)
    held_out = held_out["validation"] | held_out["test"]

    start = state[state["Property_ID"].isin(held_out)].sort_values("Prediction_Time").groupby("Property_ID").head(1)
    final_counts = assets.groupby(["Property_ID", "Category"]).size()
    rows = []
    for _, row in start.iterrows():
        features = {c: row[c] for c in state.columns if c not in NOT_FEATURES}
        start_items = int(features["documented_asset_count"])
        accepted, stages, ended = 0, [], "max steps"
        for _ in range(MAX_STEPS):
            pred = predict_next_categories(pd.DataFrame([features]))
            s = suggest_next_item(pred["top_3"], counts_by_name(features))
            if s["done"]:
                ended = "nothing left to suggest"
                break
            have = counts_by_name(features)[s["category"]]
            truth = final_counts.get((row["Property_ID"], s["category"]), 0)
            if truth > have:  # the household really has (more of) this
                features = apply_logged_item(features, s["category"], s["suggested_value"])
                accepted += 1
                stages.append(s["stage"])
            else:
                ended = "customer said no"
                break
        rows.append({"start_items": start_items, "accepted": accepted, "items_at_end": start_items + accepted,
                     "stage1": stages.count("new_category"), "stage2": stages.count("more_of_category"), "ended": ended})

    r = pd.DataFrame(rows)
    n = len(r)
    print(f"held-out households replayed: {n} (validation + test, earliest real state of each)")
    print(f"items at the start: median {r.start_items.median():.0f}, max {r.start_items.max()}")
    print(f"\nsuggestions accepted before the first 'no' (mean {r.accepted.mean():.2f}, median {r.accepted.median():.0f}):")
    print(r.accepted.value_counts().sort_index().to_string())
    print(f"\nfirst suggestion accepted: {(r.accepted >= 1).mean():.1%}")
    print(f"loop ended because: {r.ended.value_counts().to_dict()}")
    print(f"accepted new-category suggestions: {r.stage1.sum()} | accepted 'more of a category' suggestions: {r.stage2.sum()}")


if __name__ == "__main__":
    main()
