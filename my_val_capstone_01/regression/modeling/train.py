"""Household contents-value regression: training.

Fits XGBoost and LightGBM on the log of the household total, chooses the number of trees on the
validation set (notebook 03_ky_modelling_xgboost_lightgbm) and learns the blend weight and the
70% margins on the validation set (notebook 04_ky_ensemble_and_intervals). The test set is not
used here, it is scored once in notebook 05_ky_test_evaluation.
"""

from pathlib import Path

import joblib
from lightgbm import LGBMRegressor
from loguru import logger
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
import typer
from xgboost import XGBRegressor

from my_val_capstone_01.regression.config import INTERIM_DIR, MODELS_DIR, PROCESSED_DIR

app = typer.Typer()

RANDOM_STATE = 42
CURVE_SEEDS = range(8)  # seeds that are averaged to read the number of trees from the curve
TREE_STEP = 5
N_CANDIDATES = 80
CONFIDENCE = 0.70
BASKET_SIZES = [10, 20, 30, 50]


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error in percent."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return 100.0 * np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()


def make_model(kind: str, params: dict, seed: int):
    """XGBoost ('xgb') or LightGBM ('lgbm') regressor with the absolute-error objective."""
    if kind == "xgb":
        return XGBRegressor(
            **params,
            objective="reg:absoluteerror",
            tree_method="hist",
            n_jobs=1,
            random_state=seed,
        )
    return LGBMRegressor(
        **params, objective="regression_l1", n_jobs=1, random_state=seed, verbose=-1
    )


def predict_first_trees(kind: str, model, x, n_trees: int) -> np.ndarray:
    """Dollar predictions of the first n_trees trees of a fitted model."""
    if kind == "xgb":
        return np.expm1(model.predict(x, iteration_range=(0, int(n_trees))))
    return np.expm1(model.predict(x, num_iteration=int(n_trees)))


def tree_curve(kind, params, max_trees, x_train, y_train_log, x_val, y_val) -> pd.DataFrame:
    """Validation MAE using the first k trees of a model fitted on all of train (8 seeds)."""
    ks = np.arange(TREE_STEP, max_trees + 1, TREE_STEP)
    acc = np.zeros((len(CURVE_SEEDS), len(ks)))
    for si, seed in enumerate(CURVE_SEEDS):
        model = make_model(kind, {**params, "n_estimators": max_trees}, seed)
        model.fit(x_train, y_train_log)
        acc[si] = [
            mean_absolute_error(y_val, predict_first_trees(kind, model, x_val, k)) for k in ks
        ]
    return pd.DataFrame({"trees": ks, "val_MAE_mean": acc.mean(0), "val_MAE_sd": acc.std(0)})


def choose_trees(curve: pd.DataFrame) -> tuple:
    """Number of trees with the lowest mean validation MAE and the range within 1 sd of it."""
    best = curve["val_MAE_mean"].idxmin()
    limit = curve.loc[best, "val_MAE_mean"] + curve.loc[best, "val_MAE_sd"]
    within = curve.loc[curve["val_MAE_mean"] <= limit, "trees"]
    return int(curve.loc[best, "trees"]), int(within.min()), int(within.max())


def search_space(kind: str) -> dict:
    """Random search space of the 80 candidates."""
    if kind == "xgb":
        return {
            "max_depth": randint(3, 9),
            "learning_rate": uniform(0.01, 0.29),
            "n_estimators": randint(150, 700),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "min_child_weight": [1, 3, 5, 10],
            "reg_alpha": [0, 0.01, 0.1, 1],
            "reg_lambda": [0.5, 1, 2, 5],
        }
    return {
        "num_leaves": randint(15, 63),
        "max_depth": [-1, 3, 4, 5, 6, 7, 8],
        "learning_rate": uniform(0.01, 0.29),
        "n_estimators": randint(150, 700),
        "subsample": uniform(0.6, 0.4),
        "subsample_freq": [1],  # required for 'subsample' to take effect in LightGBM
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_samples": randint(5, 50),
        "reg_alpha": [0, 0.01, 0.1, 1],
        "reg_lambda": [0.5, 1, 2, 5],
    }


def fit_one_family(kind, train_df, val_df, feature_cols, results: list) -> dict:
    """Baseline, random search and tuned model of one model family, scored on validation."""
    x_train, y_train_log = train_df[feature_cols], np.log1p(train_df["true_total_value"])
    x_val, y_val = val_df[feature_cols], val_df["true_total_value"]
    baseline = (
        {"max_depth": 5, "learning_rate": 0.05}
        if kind == "xgb"
        else {
            "max_depth": -1,
            "learning_rate": 0.05,
        }
    )
    prefix = "xgb" if kind == "xgb" else "lgbm"

    def record(name, pred, extra):
        mae, error = mean_absolute_error(y_val, pred), wape(y_val, pred)
        results.append(
            {"model": name, "val_MAE": round(mae), "val_WAPE_%": round(error, 1), **extra}
        )

    curve = tree_curve(kind, baseline, 500, x_train, y_train_log, x_val, y_val)
    k_base = choose_trees(curve)[0]
    base_model = make_model(kind, {**baseline, "n_estimators": k_base}, RANDOM_STATE)
    base_model.fit(x_train, y_train_log)
    record(f"{prefix}_baseline_log", np.expm1(base_model.predict(x_val)), {"trees": k_base})

    # hyperparameter search: the train rows first, then the validation rows, one split only
    search_df = pd.concat([train_df, val_df], ignore_index=True)
    cv_splits = [(np.arange(len(train_df)), np.arange(len(train_df), len(search_df)))]
    base_estimator = make_model(kind, {}, RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=search_space(kind),
        n_iter=N_CANDIDATES,
        scoring="neg_mean_absolute_error",
        cv=cv_splits,
        refit=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(search_df[feature_cols], np.log1p(search_df["true_total_value"]))
    best = dict(search.best_params_)
    max_trees = int(best["n_estimators"])
    tuned_curve = tree_curve(
        kind,
        {k: v for k, v in best.items() if k != "n_estimators"},
        max_trees,
        x_train,
        y_train_log,
        x_val,
        y_val,
    )
    k_tuned = choose_trees(tuned_curve)[0]
    tuned = make_model(kind, {**best, "n_estimators": k_tuned}, RANDOM_STATE)
    tuned.fit(x_train, y_train_log)
    record(
        f"{prefix}_log_tuned",
        np.expm1(tuned.predict(x_val)),
        {
            "search_val_MAE_logscale": round(-search.best_score_, 4),
            "searched_max_trees": max_trees,
            "trees": k_tuned,
        },
    )
    return {
        "model": tuned,
        "curve": tuned_curve,
        "hyperparameters": {**best, "n_estimators": k_tuned, "searched_max_trees": max_trees},
    }


def fit_regression_models(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Fit both model families on train and choose their hyperparameters on validation."""
    feature_cols = [c for c in train_df.columns if c not in ("Property_ID", "true_total_value")]
    results: list = []
    xgb = fit_one_family("xgb", train_df, val_df, feature_cols, results)
    lgbm = fit_one_family("lgbm", train_df, val_df, feature_cols, results)
    # the results table lists the XGBoost baseline, XGBoost tuned, LightGBM baseline, LightGBM tuned
    order = ["xgb_baseline_log", "xgb_log_tuned", "lgbm_baseline_log", "lgbm_log_tuned"]
    results = sorted(results, key=lambda row: order.index(row["model"]))
    validation_predictions = val_df[["Property_ID", "n_documented"]].copy()
    validation_predictions["split"] = "val"
    validation_predictions["true_total_value"] = val_df["true_total_value"].to_numpy()
    validation_predictions["xgb_tuned_pred"] = np.expm1(xgb["model"].predict(val_df[feature_cols]))
    validation_predictions["lgbm_tuned_pred"] = np.expm1(
        lgbm["model"].predict(val_df[feature_cols])
    )
    return {
        "xgb": xgb,
        "lgbm": lgbm,
        "results": pd.DataFrame(results).set_index("model"),
        "validation_predictions": validation_predictions,
    }


def fit_blend_and_margins(validation_predictions: pd.DataFrame) -> pd.DataFrame:
    """Blend weight (lowest validation MAE) and one 70% margin per basket size N."""
    val = validation_predictions
    weights = np.round(np.arange(0, 1.0001, 0.02), 2)
    curve = pd.Series(
        [
            mean_absolute_error(
                val["true_total_value"],
                w * val["xgb_tuned_pred"] + (1 - w) * val["lgbm_tuned_pred"],
            )
            for w in weights
        ],
        index=weights,
    )
    best_w = float(curve.idxmin())
    ensemble = best_w * val["xgb_tuned_pred"] + (1 - best_w) * val["lgbm_tuned_pred"]
    relative_residual = (val["true_total_value"] - ensemble).abs() / ensemble.clip(lower=1)
    items = [{"item": "blend_weight_xgboost", "value": best_w}]
    for n in sorted(val["n_documented"].unique(), key=lambda k: f"N={int(k)}"):
        in_group = val["n_documented"] == n
        margin = float(np.quantile(relative_residual[in_group], CONFIDENCE))
        items.append({"item": f"margin_N={int(n)}", "value": margin})
    items.append({"item": "confidence_level", "value": CONFIDENCE})
    items.append({"item": "pool_n30_n50", "value": 0.0})
    return pd.DataFrame(items)


@app.command()
def main(
    processed_dir: Path = PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
    interim_dir: Path = INTERIM_DIR,
):
    """Train the two models and save them with the frozen blend weight and margins."""
    logger.info("Training the regression models...")
    train_df = pd.read_parquet(processed_dir / "train_df.parquet")
    val_df = pd.read_parquet(processed_dir / "val_df.parquet")
    fitted = fit_regression_models(train_df, val_df)
    frozen = fit_blend_and_margins(fitted["validation_predictions"])

    for directory in (models_dir, interim_dir):
        directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted["xgb"]["model"], models_dir / "xgb_log_model.joblib")
    joblib.dump(fitted["lgbm"]["model"], models_dir / "lgbm_log_model.joblib")
    joblib.dump(dict(zip(frozen["item"], frozen["value"])), models_dir / "ensemble_weight_and_margins.joblib")
    hyperparameters = pd.DataFrame(
        [
            {"model": "xgboost_log_tuned", **fitted["xgb"]["hyperparameters"]},
            {"model": "lightgbm_log_tuned", **fitted["lgbm"]["hyperparameters"]},
        ]
    )
    joblib.dump(
        {
            r["model"]: {
                k: (int(v) if isinstance(v, float) and v.is_integer() else v)
                for k, v in r.items()
                if k != "model" and pd.notna(v)
            }
            for r in hyperparameters.to_dict("records")
        },
        models_dir / "final_hyperparameters.joblib",
    )
    fitted["validation_predictions"].to_parquet(
        interim_dir / "tuned_model_predictions.parquet", index=False
    )
    logger.success("Models, blend weight and margins saved.")


if __name__ == "__main__":
    app()
