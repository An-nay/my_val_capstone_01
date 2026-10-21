# -------------------------------------------------------------------
# Next-item suggestion loop
# -------------------------------------------------------------------
#
# Turns the recommender's output into one question at a time:
#   "We think you have <category> worth about $X that you haven't logged. Log it?"
# If the customer says yes the item joins their inventory, the recommender runs again on the new
# state, and the next question follows. The loop ends when the customer says no or there is nothing
# left to suggest.
#
# Two stages, because the recommender only ever suggests categories the household has not documented
# yet (it was trained to predict the next undocumented category), and there are only six:
#   1. new category      the highest-probability category with no documented item (the trained model)
#   2. more of one you   when every category has an item, the category furthest below its typical item
#      already have      count (data rule, not the trained model)
#
# The suggested value is a median from the training households, not a model prediction (see README,
# "Why the suggested value is a median"):
#   new category        the median value of the FIRST item people logged in that category
#   more of a category  the median value of all items in that category
# The customer can change the value before logging.
# All numbers come from the dataset via modeling/build_notify_settings.py.
# The sentence wording is ours: the dataset has no notification of this kind.
# -------------------------------------------------------------------

from my_val_capstone_01.modeling.settings import SETTINGS

CONTENTS = SETTINGS["item_categories"]  # household contents only, no Vehicles
TYPICAL_VALUE = SETTINGS["typical_item_value"]  # all items in the category
TYPICAL_FIRST_VALUE = SETTINGS["typical_first_item_value"]  # the first item logged in the category
TYPICAL_COUNT = SETTINGS["typical_item_count"]


def suggest_with_fallback(top_3: list, counts: dict, full_ranking_fn) -> dict:
    """
    Read the recommender's top 3 first. Only when none of the three can be suggested (all already documented, or
    Vehicles) while another contents category is still empty, read the full ranking instead.

    ``full_ranking_fn`` is called with no arguments and returns the full sorted list; it is only called when needed.
    """
    suggestion = suggest_next_item(top_3, counts)
    if suggestion.get("stage") == "new_category":
        return suggestion
    if any(counts.get(category, 0) == 0 for category in CONTENTS):  # an empty category the top 3 missed
        fallback = suggest_next_item(full_ranking_fn(), counts)
        if fallback.get("stage") == "new_category":
            return fallback
    return suggestion


def suggest_next_item(ranked: list, counts: dict) -> dict:
    """
    Pick the next item to ask about.

    Parameters
    ----------
    ranked : list
        The recommender's top 3 (predict_next_categories()["top_3"]): {"category", "probability"}, most likely first.
        If all three are already documented (or Vehicles) the loop moves on to stage 2.
    counts : dict
        Documented item count per category name (see state_features.counts_by_name).

    Returns
    -------
    dict
        ``done`` False with ``category``, ``suggested_value``, ``stage`` and ``message``,
        or ``done`` True with a ``message`` when there is nothing left to suggest.
    """
    # stage 1: a category with no item yet, best probability first
    for row in ranked:
        category = row["category"]
        if category in CONTENTS and counts.get(category, 0) == 0:
            value = TYPICAL_FIRST_VALUE[category]
            return {
                "done": False,
                "stage": "new_category",
                "category": category,
                "probability": row["probability"],
                "suggested_value": value,
                "have": 0,
                "message": f"We think you have an item in {category} worth about A${value:,.0f} that you haven't logged yet. Would you like to log it?",
            }

    # stage 2: every category has an item; ask for more where the household is furthest below typical
    shortfall = {c: TYPICAL_COUNT[c] - counts.get(c, 0) for c in CONTENTS}
    category = max(shortfall, key=shortfall.get)
    if shortfall[category] > 0:
        value = TYPICAL_VALUE[category]
        return {
            "done": False,
            "stage": "more_of_category",
            "category": category,
            "probability": None,
            "suggested_value": value,
            "have": int(counts.get(category, 0)),
            "typical": TYPICAL_COUNT[category],
            "message": f"We think you have an item in {category} worth about A${value:,.0f} that you haven't logged yet. Would you like to log it?",
        }
    return {"done": True, "message": "You've logged the usual number of items in every category. Nothing more to suggest."}

