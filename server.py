"""myVal recommender web app.

Run from the project root:  uv run uvicorn server:app --reload
Then open http://127.0.0.1:8000
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from fastapi import Body, FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

from my_val_capstone_01.modeling.notify import suggest_with_fallback  # noqa: E402
from my_val_capstone_01.modeling.ranking import full_ranking  # noqa: E402
from my_val_capstone_01.modeling.predict import predict_next_categories  # noqa: E402
from my_val_capstone_01.modeling.value_cap import apply_value_cap, regression_info, value_cap  # noqa: E402
from my_val_capstone_01.modeling.state_features import AI_SYNCED, DERIVED, apply_logged_item, counts_by_name  # noqa: E402

app = FastAPI(title="myVal Next-Category Recommender")
CATS = [f"CAT-00{i}" for i in range(1, 8)]
HOUSEHOLDS = ROOT / "models" / "demo_households.json"  # built from the dataset by modeling/build_demo_households.py


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/api/households")
def households():
    """The example households, dropdown options and category names the page shows, all taken from the dataset."""
    with open(HOUSEHOLDS) as handle:
        data = json.load(handle)
    data["regression"] = regression_info()  # the regression's minimum basket and interval level, read from the regression itself
    return data


@app.post("/api/predict")
def predict(payload: dict = Body(...)):
    row = {k: (None if v == "" else v) for k, v in payload.items() if k != "items"}  # a blank field in the dataset stays blank
    items = payload.get("items")  # the logged items (category, value), used only for the regression's stopping limit
    try:
        # Totals are derived from the per-category counts, as in the training data.
        doc = [float(row.get(f"documented_count_{c}", 0)) for c in CATS]
        ai = [float(row.get(f"ai_count_{c}", 0)) for c in CATS]
        row["documented_asset_count"] = sum(doc)
        row["documented_category_count"] = sum(d > 0 for d in doc)
        row["ai_asset_count"] = sum(ai)
        features = pd.DataFrame([row])
        prediction = predict_next_categories(features)
        # The next-item question, fed by the recommender's top 3 and the household's documented counts.
        # The recommender's top 3 first; the full ranking only if none of the three can be suggested.
        suggestion = suggest_with_fallback(prediction["top_3"], counts_by_name(row), lambda: full_ranking(features))
        # Stop when the next item would take the household past its estimated total value (needs 10+ valued items).
        cap = value_cap(items, row) if items else None
        prediction["suggestion"] = apply_value_cap(suggestion, cap)
        prediction["value_cap"] = cap
        return prediction
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"{type(exc).__name__}: {exc}")


@app.post("/api/log-item")
def log_item(payload: dict = Body(...)):
    """The customer said yes to a suggestion: return the household features with that item added."""
    try:
        updated = apply_logged_item(payload["features"], payload["category"], payload["value"])
        return {"features": {k: updated[k] for k in DERIVED + AI_SYNCED}}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"{type(exc).__name__}: {exc}")
