"""Household contents-value regression: inference.

Blends the two saved models (XGBoost and LightGBM, trained on log1p of the household total) with
the frozen blend weight and returns the estimate together with its 70% interval.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from my_val_capstone_01.regression.config import MODELS_DIR
from my_val_capstone_01.regression.features import CHECKPOINTS, floor_to_checkpoint, load_regression_models

NON_MODEL_COLUMNS = [
    "Property_ID",
    "true_total_value",
]


@lru_cache(maxsize=1)
def _load_artefacts() -> tuple:
    return load_regression_models(MODELS_DIR)


def checkpoint_for(n_documented_items: int) -> int:
    """Largest basket size (10, 20, 30 or 50) that the documented items reach, or 0 below 10."""
    return floor_to_checkpoint(n_documented_items)


def predict_household_value(feature_row: pd.DataFrame) -> dict:
    """
    Estimate the total contents value of a household.

    Parameters
    ----------
    feature_row : pd.DataFrame
        A single-row DataFrame with the same 26 basket features used in training, including
        ``n_documented`` (10, 20, 30 or 50) and ``sum_value`` (see features.featurize_sample).

    Returns
    -------
    dict
        Blended estimate, the two model predictions and the 70% interval of the basket size,
        all in dollars.
    """
    if len(feature_row) != 1:
        raise ValueError("feature_row must contain exactly one row.")

    xgb_model, lgbm_model, frozen = _load_artefacts()

    n_documented = int(feature_row["n_documented"].iloc[0])
    margin_key = f"margin_N={n_documented}"
    if margin_key not in frozen.index:
        raise ValueError(f"n_documented must be one of {CHECKPOINTS}, got {n_documented}.")

    # remove identifiers and the target, and keep the column order of the training data
    model_input = feature_row.drop(columns=NON_MODEL_COLUMNS, errors="ignore")
    model_input = model_input[list(xgb_model.feature_names_in_)]

    xgb_value = float(np.expm1(xgb_model.predict(model_input))[0])
    lgbm_value = float(np.expm1(lgbm_model.predict(model_input))[0])

    weight = float(frozen["blend_weight_xgboost"])
    estimate = weight * xgb_value + (1 - weight) * lgbm_value

    margin = float(frozen[margin_key])
    lower = estimate * (1 - margin)
    upper = estimate * (1 + margin)

    # the total can never be below what is already documented
    documented_value = float(feature_row["sum_value"].iloc[0])
    lower_floored = max(lower, documented_value)

    return {
        "n_documented": n_documented,
        "documented_value": round(documented_value, 2),
        "estimate": round(estimate, 2),
        "xgboost_estimate": round(xgb_value, 2),
        "lightgbm_estimate": round(lgbm_value, 2),
        "blend_weight_xgboost": round(weight, 4),
        "confidence_level": float(frozen["confidence_level"]),
        "margin": round(margin, 4),
        "lower": round(lower, 2),
        "lower_floored": round(lower_floored, 2),
        "upper": round(upper, 2),
    }
