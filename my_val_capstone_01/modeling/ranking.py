# -------------------------------------------------------------------
# Full category ranking, used only as a fallback by the suggestion loop
# -------------------------------------------------------------------
#
# predict.py returns only the top 3 of the ranking model. The suggestion normally reads that top 3. In the rare
# case where none of the three can be suggested (all already documented, or Vehicles) while another category is
# still empty, the loop needs the rest of the list. This does the same calculation as the "Ranked Recommendations"
# block of predict_next_categories (same model, same inputs, same sort) and returns every category.
# predict.py itself is unchanged from GitHub.
# -------------------------------------------------------------------

import pandas as pd

from my_val_capstone_01.modeling.predict import NON_MODEL_COLUMNS, ranking_model


def full_ranking(feature_row: pd.DataFrame) -> list:
    """All categories, most likely first, as {"category", "probability"} (probability rounded to 4 decimals)."""
    if len(feature_row) != 1:
        raise ValueError("feature_row must contain exactly one row.")
    model_input = feature_row.drop(columns=NON_MODEL_COLUMNS, errors="ignore").copy()
    probabilities = ranking_model.predict_proba(model_input)[0]
    classes = ranking_model.named_steps["classifier"].classes_
    ranked_results = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
    return [{"category": str(category), "probability": round(float(probability), 4)} for category, probability in ranked_results]
