# -------------------------------------------------------------------
# Update the recommender's household features when an item is logged
# -------------------------------------------------------------------
#
# The recommender was trained on household "states" built from documented assets. When a customer logs
# a new item, the state changes, so the recommender can be run again on the new state. This recomputes
# the features that come from documented items:
#   documented_count_<CAT>, documented_asset_count, documented_category_count,
#   documented_value_total, documented_replacement_cost_total, mean_evidence_rating
# Replaying the real assets of real households one at a time reproduces these features exactly
# (see tests/test_state_features.py).
#
# The AI-analysis features (ai_*) cannot be rebuilt exactly, because a newly logged item has not been
# through the AI. They are kept in step instead, assuming the AI would estimate the value the customer
# confirmed: ai_count_<CAT> and ai_asset_count go up by one, and ai_estimated_value_total / _mean / _max
# are updated with that value (in the data the mean is always the total divided by ai_asset_count).
# ai_category_match_rate and ai_suggestion_acceptance_rate are rates and stay as they were.
# Why: left frozen, documented_asset_count outgrows ai_asset_count after a few items, a gap the
# recommender never saw in training (documented minus AI is 0 to -2 there). On 51 real held-out
# transitions the recommender's top pick matches the one from the real next state in 88% of cases with
# frozen AI features and in 100% with them kept in step (the real item value was used for that check).
# -------------------------------------------------------------------

from my_val_capstone_01.modeling.settings import SETTINGS

CATEGORY_ID = SETTINGS["category_ids"]  # name -> CAT-00x
CATEGORY_NAME = {v: k for k, v in CATEGORY_ID.items()}
CATEGORY_IDS = list(CATEGORY_ID.values())
RATIO = SETTINGS["value_to_replacement_ratio"]  # median value / replacement cost per category
NEW_ITEM_RATING = SETTINGS["new_item_evidence_rating"]  # typical rating of an asset with no attachment

AI_SYNCED = (
    [f"ai_count_{c}" for c in CATEGORY_IDS]
    + ["ai_asset_count", "ai_estimated_value_total", "ai_estimated_value_mean", "ai_estimated_value_max"]
)

DERIVED = (
    [f"documented_count_{c}" for c in CATEGORY_IDS]
    + ["documented_asset_count", "documented_category_count", "documented_value_total",
       "documented_replacement_cost_total", "mean_evidence_rating"]
)


def _number(value, default=0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if number != number else number


def apply_logged_item(features: dict, category: str, value, replacement=None, rating=None) -> dict:
    """
    Return a copy of ``features`` with one logged item added.

    ``replacement`` and ``rating`` are only known for items that already exist in the data. For a newly
    logged item they are estimated: the replacement cost from the value and the category's typical
    value-to-replacement ratio, the rating from what an asset with no attachment typically scores.
    """
    cid = CATEGORY_ID[category]
    value = _number(value)
    if replacement is None:
        replacement = value / (RATIO.get(category) or 1.0)
    if rating is None:
        rating = NEW_ITEM_RATING

    out = dict(features)
    n = sum(_number(features.get(f"documented_count_{c}")) for c in CATEGORY_IDS)
    out[f"documented_count_{cid}"] = _number(features.get(f"documented_count_{cid}")) + 1
    out["documented_asset_count"] = n + 1
    out["documented_category_count"] = sum(1 for c in CATEGORY_IDS if _number(out.get(f"documented_count_{c}")) > 0)
    out["documented_value_total"] = _number(features.get("documented_value_total")) + value
    out["documented_replacement_cost_total"] = _number(features.get("documented_replacement_cost_total")) + _number(replacement)
    out["mean_evidence_rating"] = (_number(features.get("mean_evidence_rating")) * n + _number(rating)) / (n + 1)

    # AI features kept in step: the item is assumed to be analysed with the value the customer confirmed.
    out[f"ai_count_{cid}"] = _number(features.get(f"ai_count_{cid}")) + 1
    ai_assets = sum(_number(out.get(f"ai_count_{c}")) for c in CATEGORY_IDS)
    out["ai_asset_count"] = ai_assets
    out["ai_estimated_value_total"] = _number(features.get("ai_estimated_value_total")) + value
    out["ai_estimated_value_mean"] = out["ai_estimated_value_total"] / ai_assets
    out["ai_estimated_value_max"] = max(_number(features.get("ai_estimated_value_max")), value)
    return out


def counts_by_name(features: dict) -> dict:
    """Documented item count per category name, read from the features."""
    return {CATEGORY_NAME[c]: _number(features.get(f"documented_count_{c}")) for c in CATEGORY_IDS}
