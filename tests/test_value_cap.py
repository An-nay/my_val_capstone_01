import json

import pytest

from my_val_capstone_01.config import MODELS_DIR
from my_val_capstone_01.modeling.value_cap import apply_value_cap, logged_value, regression_info, value_cap

HOUSEHOLDS = MODELS_DIR / "demo_households.json"
needs_models = pytest.mark.skipif(
    not (HOUSEHOLDS.exists() and (MODELS_DIR / "regression" / "xgb_log_model.joblib").exists()),
    reason="needs models/demo_households.json (build_demo_households.py) and the regression models",
)
CAP = {"estimate": 28500.0, "logged": 28000.0, "headroom": 500.0}


def test_logged_value_ignores_vehicles_and_items_without_a_value():
    items = [
        {"category": "Furniture", "value": 400.0},
        {"category": "Vehicles", "value": 15000.0},
        {"category": "Electronics", "value": None},
        {"category": "Appliances", "value": 600},
    ]
    assert logged_value(items) == pytest.approx(1000.0)


def test_suggestion_above_the_headroom_stops_the_loop():
    stopped = apply_value_cap({"done": False, "suggested_value": 750.0}, CAP)
    assert stopped["done"] is True and stopped["stage"] == "value_cap"


def test_suggestion_within_the_headroom_is_kept():
    suggestion = {"done": False, "suggested_value": 500.0, "category": "Clothings"}
    assert apply_value_cap(suggestion, CAP) is suggestion


def test_no_cap_leaves_the_suggestion_alone():
    suggestion = {"done": False, "suggested_value": 99999.0}
    assert apply_value_cap(suggestion, None) is suggestion


@needs_models
def test_regression_needs_ten_valued_items_so_small_households_have_no_cap():
    with open(HOUSEHOLDS) as handle:
        households = json.load(handle)["households"]
    small = households["Typical household"]  # 7 items
    assert value_cap(small["items"], small["features"]) is None


@needs_models
def test_cap_for_a_household_with_enough_items_and_the_stopping_example():
    with open(HOUSEHOLDS) as handle:
        household = json.load(handle)["households"]["Well documented"]  # 21 items
    cap = value_cap(household["items"], household["features"])
    assert cap is not None and cap["n_documented"] == 20
    assert cap["headroom"] == pytest.approx(cap["estimate"] - cap["logged"])
    # log one more item that leaves 500 of headroom: a 750 suggestion must stop, a 300 one must not
    padded = household["items"] + [{"category": "Furniture", "value": cap["headroom"] - 500.0}]
    tight = value_cap(padded, household["features"])
    assert tight["headroom"] == pytest.approx(500.0)
    assert apply_value_cap({"done": False, "suggested_value": 750.0}, tight)["done"] is True
    assert apply_value_cap({"done": False, "suggested_value": 300.0}, tight)["done"] is False


@needs_models
def test_server_stops_the_loop_only_when_items_are_sent():
    from fastapi.testclient import TestClient

    import server

    client = TestClient(server.app)
    with open(HOUSEHOLDS) as handle:
        household = json.load(handle)["households"]["Well documented"]
    features, items = household["features"], household["items"]
    plain = client.post("/api/predict", json=features).json()
    assert plain["suggestion"].get("stage") != "value_cap" and plain["value_cap"] is None

    cap = value_cap(items, features)
    padded = items + [{"category": "Furniture", "value": cap["headroom"] - 1.0}]
    capped = client.post("/api/predict", json={**features, "items": padded}).json()
    assert capped["suggestion"]["done"] is True and capped["suggestion"]["stage"] == "value_cap"
    assert capped["value_cap"]["headroom"] == pytest.approx(1.0)


def test_regression_info_comes_from_the_regression_itself():
    info = regression_info()
    assert info is not None
    assert info["min_items"] == 10 and info["confidence_level"] == pytest.approx(0.7)
