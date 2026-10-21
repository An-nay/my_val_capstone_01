"""Household contents-value regression: production prediction pipeline.

Raw documented items and property details in, estimate and 70% interval of the total contents value out.
The models, blend weight and margins are read from models/regression/ and are never changed here.
Written by notebooks/regression/08_ky_reg_prediction_pipeline.ipynb.
"""

import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_val_capstone_01.regression.config import MODELS_DIR  # noqa: E402
from my_val_capstone_01.regression.features import (  # noqa: E402
    CHECKPOINTS,
    basket_features,
    floor_to_checkpoint,
    load_regression_models,
)
from my_val_capstone_01.regression.modeling.predict import predict_household_value  # noqa: E402

CONTENTS_CATEGORIES = [
    "Appliances",
    "Clothings",
    "Electronics",
    "Furniture",
    "Jewellery and Personal Items",
    "Other Items",
]
EXCLUDED_CATEGORIES = ["Vehicles"]
ITEM_FIELDS = ["Category", "Current_Estimated_Value"]
PROPERTY_FIELDS = [
    "Property_Type",
    "Occupancy_Type",
    "Number_Of_Occupants",
    "Number_Of_Bedrooms",
    "Property_Valuation_AUD",
]


@lru_cache(maxsize=1)
def _feature_columns() -> tuple:
    xgb_model, _, _ = load_regression_models(MODELS_DIR)
    return tuple(xgb_model.feature_names_in_)


def _known_levels(prefix: str) -> list:
    return [c[len(prefix):] for c in _feature_columns() if c.startswith(prefix)]


def predict_from_features(feature_row: pd.DataFrame) -> dict:
    """Estimate from one row of model features (see my_val_capstone_01.regression.features)."""
    return predict_household_value(feature_row)


def predict_from_items(items, property_info: dict) -> dict:
    """
    Estimate the total contents value of a household from its documented items.

    Parameters
    ----------
    items : list of dict or pd.DataFrame
        One entry per documented item with ``Category`` and ``Current_Estimated_Value``.
        The first N items (N = 10, 20, 30 or 50, the largest size reached) form the basket.
    property_info : dict
        ``Property_Type``, ``Occupancy_Type``, ``Number_Of_Occupants``,
        ``Number_Of_Bedrooms`` and ``Property_Valuation_AUD``.

    Returns
    -------
    dict
        Estimate, the two model estimates and the 70% interval, all in dollars.
    """
    items_df = pd.DataFrame(items)
    missing = [c for c in ITEM_FIELDS if c not in items_df.columns]
    if missing:
        raise ValueError(f"items are missing the fields {missing}.")
    missing = [f for f in PROPERTY_FIELDS if f not in property_info]
    if missing:
        raise ValueError(f"property_info is missing the fields {missing}.")

    # 1 validate: vehicles are not contents, and an item without a value cannot be summed
    items_df = items_df[~items_df["Category"].isin(EXCLUDED_CATEGORIES)]
    items_df = items_df[items_df["Current_Estimated_Value"].notna()].reset_index(drop=True)
    for field, prefix in [("Property_Type", "property_type_"), ("Occupancy_Type", "occupancy_type_")]:
        if property_info[field] not in _known_levels(prefix):
            raise ValueError(f"{field} must be one of {_known_levels(prefix)}, got {property_info[field]!r}.")

    # 2 checkpoint
    n_documented = floor_to_checkpoint(len(items_df))
    if n_documented == 0:
        raise ValueError(f"at least {CHECKPOINTS[0]} valued contents items are needed, got {len(items_df)}.")
    basket = items_df.iloc[:n_documented]

    # 3 features, in the column order of the saved models
    feats = basket_features(basket, pd.Series(property_info), list(_feature_columns()), CONTENTS_CATEGORIES)
    feature_row = pd.DataFrame([feats])[list(_feature_columns())]

    # 4 and 5 predict, blend and interval
    return predict_household_value(feature_row)


if __name__ == "__main__":
    demo_items = [{"Category": CONTENTS_CATEGORIES[i % 6], "Current_Estimated_Value": 500.0 + 150 * i} for i in range(20)]
    demo_property = {
        "Property_Type": "house",
        "Occupancy_Type": "owner",
        "Number_Of_Occupants": 3,
        "Number_Of_Bedrooms": 3,
        "Property_Valuation_AUD": 900000,
    }
    print(predict_from_items(demo_items, demo_property))
