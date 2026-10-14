# -------------------------------------------------------------------
# Stop the suggestion loop when the household is already near its estimated total value
# -------------------------------------------------------------------
#
# Uses the household contents-value regression (the committed models in models/regression/, never
# retrained or changed here) to estimate what the whole household is worth, and compares it with what the
# customer has already logged:
#
#   headroom = estimate - value already logged      (never below 0)
#
# If the next suggested item is worth more than the headroom, the loop stops. The regression needs at least
# 10 valued, non-vehicle items, so for smaller households (and on any error) there is no cap and the loop
# behaves exactly as it does without this module.
# -------------------------------------------------------------------

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROPERTY_FIELDS = ["Property_Type", "Occupancy_Type", "Number_Of_Occupants", "Number_Of_Bedrooms", "Property_Valuation_AUD"]
EXCLUDED_CATEGORIES = ("Vehicles",)  # not household contents, so not part of the regression's total

_predict_from_items = None


def _regression():
    """Load the regression pipeline once, on first use (it loads the saved models)."""
    global _predict_from_items
    if _predict_from_items is None:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from modeling.reg_predict_pipeline import predict_from_items

        _predict_from_items = predict_from_items
    return _predict_from_items


def regression_info():
    """The regression's own settings, so the page does not hard-code them: the smallest basket and the interval level."""
    try:
        from my_val_capstone_01.regression.features import CHECKPOINTS, load_regression_models

        frozen = load_regression_models()[2]
        return {"min_items": int(CHECKPOINTS[0]), "confidence_level": float(frozen["confidence_level"])}
    except Exception:
        return None


def logged_value(items) -> float:
    """Value already logged: the valued, non-vehicle items (an item with no recorded value adds nothing)."""
    total = 0.0
    for item in items or []:
        value = item.get("value")
        if value is not None and value == value and item.get("category") not in EXCLUDED_CATEGORIES:
            total += float(value)
    return total


def value_cap(items, features: dict):
    """
    The household's estimated total value and the headroom left, or None when the regression cannot be used.

    Parameters
    ----------
    items : list of dict
        The logged items, oldest first, each with ``category`` and ``value`` (None if the value is not recorded).
    features : dict
        The household's features; only the five property fields of the regression are read.
    """
    try:
        regression_items = [
            {"Category": item["category"], "Current_Estimated_Value": item.get("value")}
            for item in items or []
        ]
        property_info = {field: features[field] for field in PROPERTY_FIELDS}
        result = _regression()(regression_items, property_info)
    except Exception:  # fewer than 10 valued items, a missing property field, an unknown level, ...
        return None
    logged = logged_value(items)
    estimate = float(result["estimate"])
    return {
        "estimate": round(estimate, 2),
        "lower": float(result["lower_floored"]),
        "upper": float(result["upper"]),
        "logged": round(logged, 2),
        "headroom": round(max(estimate - logged, 0.0), 2),
        "n_documented": int(result["n_documented"]),
    }


def apply_value_cap(suggestion: dict, cap):
    """Replace the suggestion by a 'done' message when it is worth more than the headroom."""
    if not cap or suggestion.get("done") or suggestion["suggested_value"] <= cap["headroom"]:
        return suggestion
    return {
        "done": True,
        "stage": "value_cap",
        "message": (
            f"The items you have logged (A${cap['logged']:,.0f}) already come to about what we would expect for a home like "
            f"yours (about A${cap['estimate']:,.0f}). Nothing more to suggest."
        ),
    }
