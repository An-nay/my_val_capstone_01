import pandas as pd

from my_val_capstone_01.modeling.predict import predict_next_categories
from my_val_capstone_01.modeling.ranking import full_ranking

CATS = [f"CAT-00{i}" for i in range(1, 8)]


def _row():
    row = {
        "mean_evidence_rating": 2, "documented_value_total": 3000, "documented_replacement_cost_total": 3300,
        "ai_category_match_rate": 1, "ai_suggestion_acceptance_rate": 0.5, "ai_estimated_value_total": 2300,
        "ai_estimated_value_mean": 760, "ai_estimated_value_max": 1300, "Property_Type": "house",
        "Occupancy_Type": "owner", "Number_Of_Occupants": 3, "Number_Of_Bedrooms": 4,
        "Property_Valuation_AUD": 109292, "Is_Home_Insured": 1, "Insurance_Coverage_Type": "home-and-contents-insurance",
        "Insurance_Confidence_1_5": 3, "Is_Primary": True, "Notification_Opt_In": True, "Tenure_Months": 20,
        "Age_Band": "35-44", "Gender": "Female", "State": "NSW", "Occupation": "Accountant",
        "Household_Type": "family_with_children", "Income_Band": "75k_100k", "Financial_Literacy_Band": "medium",
        "Engagement_Tier": "engaged", "Membership_Level": "free", "Account_Status": "active",
        "Preferred_Channel": "email", "Preferred_Explanation_Style": "short_summary", "Prior_Claim_Experience": "none",
        "Underinsurance_Awareness": "somewhat_aware", "video_count": 0, "total_video_duration_seconds": 0,
        "mean_video_duration_seconds": 0, "total_rooms_covered": 0, "total_items_detected": 0,
        "documented_asset_count": 7, "documented_category_count": 5, "ai_asset_count": 7,
    }
    for c, n in zip(CATS, [2, 1, 2, 0, 1, 1, 0]):
        row[f"documented_count_{c}"] = n
        row[f"ai_count_{c}"] = n
    return row


def test_prediction_has_top_1_top_2_and_top_3():
    result = predict_next_categories(pd.DataFrame([_row()]))
    assert {"top_1", "top_2", "top_3"} <= set(result)
    assert len(result["top_2"]) == 2 and len(result["top_3"]) == 3
    probabilities = [r["probability"] for r in result["top_3"]]
    assert probabilities == sorted(probabilities, reverse=True)


def test_full_ranking_covers_every_category_and_starts_with_the_top_3():
    frame = pd.DataFrame([_row()])
    ranked = full_ranking(frame)
    assert len(ranked) == 7
    probabilities = [r["probability"] for r in ranked]
    assert probabilities == sorted(probabilities, reverse=True)
    assert ranked[:3] == predict_next_categories(frame)["top_3"]
