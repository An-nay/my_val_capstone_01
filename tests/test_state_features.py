import pandas as pd
import pytest

from my_val_capstone_01.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from my_val_capstone_01.modeling.state_features import CATEGORY_ID, DERIVED, apply_logged_item, counts_by_name

WORKBOOK = RAW_DATA_DIR / "myVal_Synthetic_Datasets_release_v2.xlsx"
MODEL_STATE = PROCESSED_DATA_DIR / "model_state.csv"

FEATURES = {
    "documented_count_CAT-001": 1, "documented_count_CAT-002": 0, "documented_count_CAT-003": 1,
    "documented_count_CAT-004": 0, "documented_count_CAT-005": 0, "documented_count_CAT-006": 0,
    "documented_count_CAT-007": 1,
    "documented_asset_count": 3, "documented_category_count": 3, "documented_value_total": 1000.0,
    "documented_replacement_cost_total": 1150.0, "mean_evidence_rating": 3.0, "ai_asset_count": 3,
    "ai_count_CAT-001": 1, "ai_count_CAT-002": 0, "ai_count_CAT-003": 1, "ai_count_CAT-004": 0,
    "ai_count_CAT-005": 0, "ai_count_CAT-006": 0, "ai_count_CAT-007": 1,
    "ai_estimated_value_total": 900.0, "ai_estimated_value_mean": 300.0, "ai_estimated_value_max": 500.0,
    "ai_category_match_rate": 1.0, "ai_suggestion_acceptance_rate": 0.5,
}


def test_logging_an_item_moves_the_documented_features():
    out = apply_logged_item(FEATURES, "Appliances", 700.0)
    assert out["documented_count_CAT-005"] == 1 and out["documented_asset_count"] == 4
    assert out["documented_category_count"] == 4 and out["documented_value_total"] == pytest.approx(1700.0)
    assert out["documented_replacement_cost_total"] > 1150.0
    assert counts_by_name(out)["Appliances"] == 1


def test_logging_leaves_other_features_and_the_input_alone():
    out = apply_logged_item(FEATURES, "Electronics", 100.0)
    assert FEATURES["documented_asset_count"] == 3 and FEATURES["ai_asset_count"] == 3  # the input is not modified
    assert out["ai_category_match_rate"] == 1.0 and out["ai_suggestion_acceptance_rate"] == 0.5  # rates stay as they were
    assert out["documented_count_CAT-001"] == 2 and out["documented_category_count"] == 3  # category already counted


@pytest.mark.skipif(not (WORKBOOK.exists() and MODEL_STATE.exists()), reason="needs the data")
def test_logging_real_assets_one_by_one_reproduces_the_next_real_state():
    """Replay: state t plus the assets created before the next state t' equals the real state t'."""
    assets = pd.read_excel(WORKBOOK, sheet_name="assets")
    assets["Created_At"] = pd.to_datetime(assets["Created_At"], errors="coerce")
    state = pd.read_csv(MODEL_STATE, parse_dates=["Prediction_Time"]).sort_values(["Property_ID", "Prediction_Time"])
    checked = 0
    for pid, g in state.groupby("Property_ID"):
        g = g.reset_index(drop=True)
        for k in range(len(g) - 1):
            now, nxt = g.loc[k], g.loc[k + 1]
            new = assets[(assets["Property_ID"] == pid) & (assets["Created_At"] > now["Prediction_Time"]) & (assets["Created_At"] <= nxt["Prediction_Time"])]
            features = {c: now[c] for c in state.columns}
            for r in new.itertuples():
                features = apply_logged_item(
                    features, r.Category, r.Current_Estimated_Value,
                    replacement=0.0 if pd.isna(r.Replacement_Cost_New) else r.Replacement_Cost_New,
                    rating=r.Evidence_Rating_0_5,
                )
            for key in DERIVED:
                assert features[key] == pytest.approx(nxt[key], abs=1e-6), (pid, key)
            checked += 1
        if checked >= 40:
            break
    assert checked >= 40


def test_ai_features_are_kept_in_step_with_the_logged_item():
    out = apply_logged_item(FEATURES, "Appliances", 700.0)
    assert out["ai_count_CAT-005"] == 1 and out["ai_asset_count"] == 4
    assert out["ai_estimated_value_total"] == pytest.approx(1600.0)
    assert out["ai_estimated_value_mean"] == pytest.approx(1600.0 / 4)
    assert out["ai_estimated_value_max"] == pytest.approx(700.0)
    assert out["documented_asset_count"] - out["ai_asset_count"] == FEATURES["documented_asset_count"] - FEATURES["ai_asset_count"]
    again = apply_logged_item(out, "Appliances", 100.0)
    assert again["ai_asset_count"] == 5 and again["ai_estimated_value_max"] == pytest.approx(700.0)
    assert again["ai_estimated_value_mean"] == pytest.approx(1700.0 / 5)
