"""Build models/notify_settings.json: the numbers the suggestion loop uses, from the dataset.

Run from the project root (needs data/raw/... and data/processed/model_state.csv, see notebook 02):

    python modeling/build_notify_settings.py

Nothing here is a trained model. Everything that describes "typical" households or items is taken from
the TRAINING households only (the split of 03_fm_modelling), so validation and test households never
influence it. Where each value comes from:
  item_categories              base_categories names in Display_Order, without Vehicles (not household contents)
  category_ids                 base_categories name -> Category_ID (the ids the model's features use)
  typical_item_value           median assets.Current_Estimated_Value per category, over all assets
                               (the suggested value for "do you have another one?")
  typical_first_item_value     median value of the item documented at the first event of the target category,
                               over the training states (the suggested value for a NEW category)
  first_item_examples          how many training examples sit behind each first-item median
  typical_item_count           per category, the median number of items among households that have the category
  value_to_replacement_ratio   median Current_Estimated_Value / Replacement_Cost_New per category
                               (turns a logged item's value into a replacement cost for the model's feature)
  new_item_evidence_rating     median Evidence_Rating_0_5 of assets with no attachment (no proof of purchase),
                               i.e. what a newly logged item typically scores
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
OUTPUT = ROOT / "models" / "notify_settings.json"


def main(workbook: Path = WORKBOOK, output: Path = OUTPUT) -> dict:
    sheets = pd.read_excel(workbook, sheet_name=["assets", "base_categories", "asset_attachments"])
    assets, cats, attachments = sheets["assets"], sheets["base_categories"].sort_values("Display_Order"), sheets["asset_attachments"]

    state = pd.read_csv(MODEL_STATE)
    train = split_properties(state)["train"]
    tr = assets[assets["Property_ID"].isin(train)]

    contents = [n for n in cats["Name"] if n != "Vehicles"]

    # value of the item(s) created at the first event of each (household, category); median if several
    assets["Created_At"] = pd.to_datetime(assets["Created_At"], errors="coerce")
    first = assets.groupby(["Property_ID", "Category"])["Created_At"].min().rename("t").reset_index()
    events = assets.merge(first, on=["Property_ID", "Category"])
    events = events[events["Created_At"] == events["t"]]
    first_value = events.groupby(["Property_ID", "Category"])["Current_Estimated_Value"].median().rename("first_value").reset_index()
    targets = state[state["Property_ID"].isin(train)].merge(first_value, left_on=["Property_ID", "Next_Category"], right_on=["Property_ID", "Category"], how="left")
    targets = targets[(targets["first_value"] > 0) & targets["Next_Category"].isin(contents)]
    first_item = targets.groupby("Next_Category")["first_value"].agg(["median", "count"])
    per_property = tr.groupby(["Property_ID", "Category"]).size()
    ratio = (tr["Current_Estimated_Value"] / tr["Replacement_Cost_New"]).replace([np.inf, -np.inf], np.nan)
    ratio = ratio.groupby(tr["Category"]).median()
    no_evidence = tr[~tr["Asset_ID"].isin(set(attachments["Asset_ID"]))]

    result = {
        "item_categories": contents,
        "category_ids": dict(zip(cats["Name"], cats["Category_ID"])),
        "typical_item_value": tr.groupby("Category")["Current_Estimated_Value"].median().round(2).to_dict(),
        "typical_first_item_value": {c: float(first_item.loc[c, "median"]) for c in contents},
        "first_item_examples": {c: int(first_item.loc[c, "count"]) for c in contents},
        "typical_item_count": {c: int(per_property[:, c].median()) for c in contents},
        "value_to_replacement_ratio": ratio.round(4).to_dict(),
        "new_item_evidence_rating": float(no_evidence["Evidence_Rating_0_5"].median()),
        "built_from": f"{len(train)} training households",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"wrote {output}  ({len(no_evidence)} training assets have no attachment)")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
