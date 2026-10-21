"""Household contents-value regression: feature construction.

Two groups of features are built here:

* the basket features that the regression models are trained on (``featurize_sample``), and
* the five optional regression columns that can be merged into the classification dataset
  (``build_regression_state_features``). This merge is not needed by any notebook of this
  repository, and the classification files are the team's originals.
"""

from pathlib import Path

import joblib
from loguru import logger
import numpy as np
import pandas as pd
import typer

from my_val_capstone_01.regression.config import (
    CLASSIFICATION_PROCESSED_DIR,
    INTERIM_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

app = typer.Typer()

CHECKPOINTS = [10, 20, 30, 50]
NEW_COLUMNS = [
    "documented_value_nv",
    "reg_pred_total_nv",
    "reg_pred_gap_nv",
    "reg_completeness_nv",
    "reg_available",
]


# -------------------------------------------------------------------
# Basket features (training data of the regression)
# -------------------------------------------------------------------


def category_column(category: str) -> str:
    """Name of the feature that holds the value documented in one item category."""
    cleaned = category.lower().replace(" ", "_").replace("(", "").replace(")", "")
    return "cat_" + cleaned.replace("/", "_")


def featurize_sample(
    sample_items: pd.DataFrame, prop_row: pd.Series, contents_categories: list
) -> dict:
    """Turn the N documented items of one home into one row of regression features."""
    values = sample_items["Current_Estimated_Value"].to_numpy(dtype=float)
    feats = {
        "item_count": len(values),
        "sum_value": values.sum(),
        "mean_value": values.mean(),
        "median_value": np.median(values),
        "max_value": values.max(),
        "min_value": values.min(),
        "std_value": values.std() if len(values) > 1 else 0.0,
        "n_categories_seen": sample_items["Category"].nunique(),
    }
    # value documented so far in each category (0 if that category was not drawn)
    category_sums = sample_items.groupby("Category")["Current_Estimated_Value"].sum()
    for category in contents_categories:
        feats[category_column(category)] = float(category_sums.get(category, 0.0))
    # household context (the same for every sample of this property)
    feats["property_type"] = prop_row["Property_Type"]
    feats["occupancy_type"] = prop_row["Occupancy_Type"]
    feats["number_of_occupants"] = prop_row["Number_Of_Occupants"]
    feats["number_of_bedrooms"] = prop_row["Number_Of_Bedrooms"]
    feats["property_valuation_aud"] = prop_row["Property_Valuation_AUD"]
    return feats


# -------------------------------------------------------------------
# Regression columns for the classification dataset
# -------------------------------------------------------------------


def floor_to_checkpoint(n_items: int) -> int:
    """Largest basket size (10, 20, 30 or 50) reached by n_items, or 0 below 10 items."""
    reached = [size for size in CHECKPOINTS if n_items >= size]
    return max(reached) if reached else 0


def basket_features(
    basket: pd.DataFrame,
    context: pd.Series,
    feature_cols: list,
    contents_categories: list,
) -> dict:
    """Regression features of the earliest-N basket of one classification state."""
    values = basket["Current_Estimated_Value"].to_numpy(dtype=float)
    feats = {
        "item_count": len(values),
        "sum_value": values.sum(),
        "mean_value": values.mean(),
        "median_value": float(np.median(values)),
        "max_value": values.max(),
        "min_value": values.min(),
        "std_value": values.std() if len(values) > 1 else 0.0,
        "n_categories_seen": basket["Category"].nunique(),
    }
    category_sums = basket.groupby("Category")["Current_Estimated_Value"].sum()
    for category in contents_categories:
        feats[category_column(category)] = float(category_sums.get(category, 0.0))
    feats["number_of_occupants"] = context["Number_Of_Occupants"]
    feats["number_of_bedrooms"] = context["Number_Of_Bedrooms"]
    feats["property_valuation_aud"] = context["Property_Valuation_AUD"]
    feats["n_documented"] = len(values)
    for col in feature_cols:
        if col.startswith("property_type_"):
            feats[col] = bool(context["Property_Type"] == col[len("property_type_") :])
        if col.startswith("occupancy_type_"):
            feats[col] = bool(context["Occupancy_Type"] == col[len("occupancy_type_") :])
    return feats


def load_regression_models(models_dir: Path = MODELS_DIR) -> tuple:
    """Load the two saved regression models and the frozen blend weight and margins."""
    xgb_model = joblib.load(models_dir / "xgb_log_model.joblib")
    lgbm_model = joblib.load(models_dir / "lgbm_log_model.joblib")
    frozen = joblib.load(models_dir / "ensemble_weight_and_margins.joblib")  # dict: blend weight, margin per N, confidence
    return xgb_model, lgbm_model, pd.Series(frozen)


def build_regression_state_features(
    model_state: pd.DataFrame,
    assets: pd.DataFrame,
    xgb_model,
    lgbm_model,
    weight_xgb: float,
    state_split: pd.Series,
) -> tuple:
    """Add the five regression columns to the classification states.

    For every state, only items documented by the state time (Created_At <= Prediction_Time)
    are used. The basket is the earliest N non-vehicle items with a value, where N is the
    number of documented items rounded down to 10, 20, 30 or 50. States with fewer than 10
    such items receive -1 in the three estimate columns. Nothing is trained here.

    Returns the extended model state and a diagnostics table with the intermediate values.
    """
    group_keys = ["Property_ID", "Prediction_Time"]
    original_columns = list(model_state.columns)
    state_time = pd.to_datetime(model_state["Prediction_Time"])
    assets = assets.copy()
    assets["Created_At"] = pd.to_datetime(assets["Created_At"], errors="coerce")
    assert not model_state.duplicated(subset=group_keys).any()
    assert assets["Created_At"].notna().all()

    keys = model_state[group_keys].copy()
    keys["_time"] = state_time.values
    history = keys.merge(
        assets[["Property_ID", "Asset_ID", "Category", "Current_Estimated_Value", "Created_At"]],
        on="Property_ID",
        how="left",
    )
    history = history[history["Created_At"] <= history["_time"]].copy()

    # the classification team's own documented count and value (vehicles included)
    everything = (
        history.groupby(group_keys)
        .agg(
            all_assets=("Asset_ID", "nunique"),
            all_value=("Current_Estimated_Value", "sum"),
        )
        .reset_index()
    )
    check = model_state[group_keys + ["documented_asset_count", "documented_value_total"]].merge(
        everything, on=group_keys, how="left"
    )
    assert (check["all_assets"] == check["documented_asset_count"]).all()
    assert np.allclose(check["all_value"], check["documented_value_total"])

    # the same assets without vehicles (and only items that have a value)
    is_vehicle = history["Category"] == "Vehicles"
    has_value = history["Current_Estimated_Value"].notna()
    non_vehicle = (
        history[~is_vehicle & has_value]
        .groupby(group_keys)
        .agg(
            n_nv=("Asset_ID", "nunique"),
            documented_value_nv=("Current_Estimated_Value", "sum"),
        )
        .reset_index()
    )
    vehicles = (
        history[is_vehicle & has_value]
        .groupby(group_keys)
        .agg(vehicle_value_recorded=("Current_Estimated_Value", "sum"))
        .reset_index()
    )
    state_values = (
        model_state[group_keys + ["documented_value_total"]]
        .merge(non_vehicle, on=group_keys, how="left")
        .merge(vehicles, on=group_keys, how="left")
        .fillna({"n_nv": 0, "documented_value_nv": 0.0, "vehicle_value_recorded": 0.0})
    )
    state_values["n_nv"] = state_values["n_nv"].astype(int)
    assert np.allclose(
        state_values["documented_value_nv"] + state_values["vehicle_value_recorded"],
        state_values["documented_value_total"],
    )
    state_values["checkpoint"] = state_values["n_nv"].map(floor_to_checkpoint)
    state_values["reg_available"] = (state_values["checkpoint"] > 0).astype(int)

    # baskets: the earliest N items of each state with an estimate
    feature_cols = [str(c) for c in xgb_model.feature_names_in_]
    assert feature_cols == [str(c) for c in lgbm_model.feature_name_]
    items = assets[(assets["Category"] != "Vehicles") & assets["Current_Estimated_Value"].notna()]
    items = items.sort_values(["Property_ID", "Created_At", "Asset_ID"])
    items_by_property = {pid: grp for pid, grp in items.groupby("Property_ID")}
    contents_categories = sorted(assets.loc[assets["Category"] != "Vehicles", "Category"].unique())

    rows, basket_info = [], []
    for i in state_values.index[state_values["reg_available"] == 1]:
        pid, state_at = model_state.at[i, "Property_ID"], state_time[i]
        checkpoint = int(state_values.at[i, "checkpoint"])
        documented = items_by_property[pid]
        documented = documented[documented["Created_At"] <= state_at]
        basket = documented.iloc[:checkpoint]
        assert len(basket) == checkpoint
        assert len(documented) == state_values.at[i, "n_nv"]
        assert floor_to_checkpoint(len(documented)) == checkpoint
        assert basket["Created_At"].max() <= state_at
        cut_has_tie = bool(
            len(documented) > checkpoint
            and documented.iloc[checkpoint - 1]["Created_At"]
            == documented.iloc[checkpoint]["Created_At"]
        )
        rows.append(basket_features(basket, model_state.loc[i], feature_cols, contents_categories))
        basket_info.append(
            {
                "index": i,
                "basket_sum": basket["Current_Estimated_Value"].sum(),
                "last_basket_time": basket["Created_At"].max(),
                "tie_at_cut": cut_has_tie,
            }
        )
    x_states = pd.DataFrame(rows, index=[b["index"] for b in basket_info])[feature_cols]
    basket_info = pd.DataFrame(basket_info).set_index("index")

    # predict, blend, clip at the documented value and pad the states without an estimate
    xgb_pred = np.expm1(xgb_model.predict(x_states))
    lgbm_pred = np.expm1(lgbm_model.predict(x_states))
    blend = weight_xgb * xgb_pred + (1 - weight_xgb) * lgbm_pred

    feature_table = state_values[
        group_keys + ["documented_value_nv", "reg_available", "checkpoint", "n_nv"]
    ].copy()
    for col, values in [("xgb_pred", xgb_pred), ("lgbm_pred", lgbm_pred), ("blend_raw", blend)]:
        feature_table[col] = np.nan
        feature_table.loc[x_states.index, col] = values
    available = feature_table["reg_available"] == 1
    feature_table["clipped"] = available & (
        feature_table["blend_raw"] < feature_table["documented_value_nv"]
    )
    total = np.maximum(feature_table["blend_raw"], feature_table["documented_value_nv"])
    feature_table["reg_pred_total_nv"] = np.where(available, total, -1.0)
    feature_table["reg_pred_gap_nv"] = np.where(
        available, total - feature_table["documented_value_nv"], -1.0
    )
    feature_table["reg_completeness_nv"] = np.where(
        available, feature_table["documented_value_nv"] / total.where(total > 0, np.nan), -1.0
    )

    final_model_state = model_state.copy()
    for col in NEW_COLUMNS:
        final_model_state[col] = feature_table[col].values
    final_model_state["reg_available"] = final_model_state["reg_available"].astype(int)
    assert final_model_state[original_columns].equals(model_state)
    assert final_model_state[NEW_COLUMNS].notna().all().all()

    split_values = np.asarray(state_split)
    diagnostics = pd.DataFrame(
        {
            "Property_ID": model_state["Property_ID"],
            "Prediction_Time": model_state["Prediction_Time"],
            "split": split_values,
            "documented_value_total": model_state["documented_value_total"],
            "documented_value_nv": feature_table["documented_value_nv"],
            "vehicle_value_recorded": state_values["vehicle_value_recorded"],
            "n_nv": feature_table["n_nv"],
            "checkpoint": feature_table["checkpoint"],
            "reg_available": final_model_state["reg_available"],
            "basket_sum": basket_info["basket_sum"].reindex(model_state.index),
            "last_basket_time": basket_info["last_basket_time"].reindex(model_state.index),
            "xgb_pred": feature_table["xgb_pred"],
            "lgbm_pred": feature_table["lgbm_pred"],
            "blend_raw": feature_table["blend_raw"],
            "clipped": feature_table["clipped"],
        }
    )
    diagnostics["pred_is_in_sample"] = (diagnostics["split"] == "train") & (
        diagnostics["reg_available"] == 1
    )
    diagnostics["reported_total_incl_vehicle"] = np.where(
        diagnostics["reg_available"] == 1,
        final_model_state["reg_pred_total_nv"] + diagnostics["vehicle_value_recorded"],
        np.nan,
    )
    return final_model_state, diagnostics


@app.command()
def main(
    model_state_path: Path = PROCESSED_DATA_DIR / "model_state.csv",
    xlsx_path: Path = RAW_DATA_DIR / "myVal_Synthetic_Datasets_release_v2.xlsx",
    models_dir: Path = MODELS_DIR,
    output_path: Path = CLASSIFICATION_PROCESSED_DIR / "model_state_with_reg.csv",
    diagnostics_path: Path = INTERIM_DIR / "regression_state_diagnostics.csv",
):
    """Optional: write model_state.csv plus the five regression columns (nothing is trained)."""
    from my_val_capstone_01.regression.dataset import rebuild_state_split

    logger.info("Merging the regression into the classification states (optional step)...")
    model_state = pd.read_csv(model_state_path, float_precision="round_trip")
    assets = pd.read_excel(xlsx_path, sheet_name="assets")
    xgb_model, lgbm_model, frozen = load_regression_models(models_dir)
    final_model_state, diagnostics = build_regression_state_features(
        model_state,
        assets,
        xgb_model,
        lgbm_model,
        float(frozen["blend_weight_xgboost"]),
        rebuild_state_split(model_state),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    final_model_state.to_csv(output_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    n_estimates = int(final_model_state["reg_available"].sum())
    logger.success(f"{len(final_model_state)} states saved, {n_estimates} with an estimate.")


if __name__ == "__main__":
    app()
