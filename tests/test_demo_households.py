import json

import pandas as pd
import pytest

from my_val_capstone_01.config import MODELS_DIR, PROCESSED_DATA_DIR

HOUSEHOLDS = MODELS_DIR / "demo_households.json"
MODEL_STATE = PROCESSED_DATA_DIR / "model_state.csv"
CATS = [f"CAT-00{i}" for i in range(1, 8)]

pytestmark = pytest.mark.skipif(not HOUSEHOLDS.exists(), reason="run modeling/build_demo_households.py first")


@pytest.fixture(scope="module")
def data():
    with open(HOUSEHOLDS) as handle:
        return json.load(handle)


def test_listed_items_reproduce_each_households_counts_and_value(data):
    for name, h in data["households"].items():
        f, items = h["features"], h["items"]
        assert f["documented_asset_count"] == len(items), name
        for c in CATS:
            assert f[f"documented_count_{c}"] == sum(i["category_id"] == c for i in items), (name, c)
        assert f["documented_value_total"] == pytest.approx(sum(i["value"] or 0 for i in items)), name


def test_every_item_has_a_real_name_and_category(data):
    for h in data["households"].values():
        for i in h["items"]:
            assert i["name"] and i["category_id"] in data["labels"]


@pytest.mark.skipif(not MODEL_STATE.exists(), reason="needs model_state.csv")
def test_households_are_real_model_state_rows_and_options_cover_them(data):
    state = pd.read_csv(MODEL_STATE)
    for name, h in data["households"].items():
        row = state[(state["Property_ID"] == h["property_id"]) & (state["Prediction_Time"] == h["prediction_time"])]
        assert len(row) == 1, name
        for column, options in data["options"].items():
            value = h["features"][column]
            assert value is None or value in options, (name, column)
    for column, options in data["options"].items():
        assert set(options) == set(state[column].dropna().astype(str)), column  # nothing typed by hand


def test_the_quoted_holdout_figures_are_in_the_file(data):
    holdout = data["holdout"]
    assert holdout["states"] > 0 and 0 < holdout["top_2"] <= holdout["top_3"] <= 1
