import os

import pandas as pd
import pytest

from my_val_capstone_01.regression.config import (
    CLASSIFICATION_PROCESSED_DIR,
    INTERIM_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROCESSED_DIR,
    RAW_DATA_DIR,
)
from my_val_capstone_01.regression.features import floor_to_checkpoint
from my_val_capstone_01.regression.modeling.predict import checkpoint_for, predict_household_value

WORKBOOK = RAW_DATA_DIR / "myVal_Synthetic_Datasets_release_v2.xlsx"
MODEL_STATE = PROCESSED_DATA_DIR / "model_state.csv"
SLOW = pytest.mark.skipif(os.environ.get("RUN_SLOW_TESTS") != "1", reason="set RUN_SLOW_TESTS=1")


def test_checkpoint_uses_the_largest_basket_reached():
    assert [floor_to_checkpoint(n) for n in (9, 10, 19, 20, 49, 80)] == [0, 10, 10, 20, 30, 50]
    assert checkpoint_for(49) == 30


def test_predict_household_value_rejects_other_basket_sizes():
    row = pd.DataFrame([{"n_documented": 15, "sum_value": 1000.0}])
    with pytest.raises(ValueError):
        predict_household_value(row)


@pytest.mark.skipif(
    not ((PROCESSED_DIR / "test_df.parquet").exists() and (INTERIM_DIR / "test_ensemble_predictions.parquet").exists()),
    reason="needs the tables and the saved test predictions of notebook 05_ky_test_evaluation",
)
def test_predictions_equal_the_saved_test_predictions():
    test_df = pd.read_parquet(PROCESSED_DIR / "test_df.parquet")
    saved = pd.read_parquet(INTERIM_DIR / "test_ensemble_predictions.parquet")
    result = pd.DataFrame([predict_household_value(test_df.iloc[[i]]) for i in range(len(test_df))])
    assert (result["estimate"] - saved["ens"]).abs().max() < 0.01
    assert (result["upper"] - saved["upper"]).abs().max() < 0.01


@pytest.mark.skipif(
    not (
        WORKBOOK.exists()
        and MODEL_STATE.exists()
        and (CLASSIFICATION_PROCESSED_DIR / "model_state_with_reg.csv").exists()
    ),
    reason="needs the data and model_state_with_reg.csv (optional, made by "
    "python -m my_val_capstone_01.regression.features)",
)
def test_state_features_equal_the_notebook_output(tmp_path):
    from my_val_capstone_01.regression.features import main

    main(
        model_state_path=MODEL_STATE,
        xlsx_path=WORKBOOK,
        models_dir=MODELS_DIR,
        output_path=tmp_path / "with_reg.csv",
        diagnostics_path=tmp_path / "diagnostics.csv",
    )
    new = pd.read_csv(tmp_path / "with_reg.csv", float_precision="round_trip")
    saved = pd.read_csv(CLASSIFICATION_PROCESSED_DIR / "model_state_with_reg.csv", float_precision="round_trip")
    assert new.equals(saved)


@SLOW
@pytest.mark.skipif(not (WORKBOOK.exists() and MODEL_STATE.exists()), reason="needs the data")
def test_modelling_tables_equal_the_notebook_output(tmp_path):
    from my_val_capstone_01.regression.dataset import main

    main(WORKBOOK, MODEL_STATE, tmp_path / "interim", tmp_path / "processed")
    for name in ("train_df", "val_df", "test_df", "unlisted_df"):
        new = pd.read_parquet(tmp_path / "processed" / f"{name}.parquet")
        assert new.equals(pd.read_parquet(PROCESSED_DIR / f"{name}.parquet"))
