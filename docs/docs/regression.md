# Household contents-value regression

This model sits next to the next-category recommendation model. It estimates the **total contents value of a household** from the N items that the customer has documented (N = 10, 20, 30 or 50) and returns a **70% interval** around the estimate. Vehicles are excluded from the regression.

## Folder layout

The regression files are kept in their own `regression/` subfolders. The files of the classification team (the notebooks `01_fm` to `05_fm`, the Random Forest models, `modeling/predict.py`, `server.py` and `index.html`) are the team's originals and are left where they are, except that `server.py` and `index.html` were later extended for the suggestion loop (see the README).

| Folder | Regression | Classification (the team's files) |
|---|---|---|
| `notebooks/` | `regression/`: `01_ky` to `07_ky`, and the regression pipeline (`08_ky`) | `01_fm` to `05_fm` |
| `data/interim/` | `regression/`: sheet copies, the property split, predictions and diagnostics | none |
| `data/processed/` | `regression/`: the modelling tables and the prediction table | `model_state.csv` (read by the regression to rebuild the split) |
| `modeling/` | `reg_predict_pipeline.py`, the production pipeline written by `08_ky` | `predict.py` |
| `models/` | `regression/`: the two models, the hyperparameters, and the blend weight and margins, all as joblib files | the Top-1 and ranking Random Forest models in the folder root |

## Regression notebooks (`notebooks/regression/`, run in this order)

| Notebook | Purpose | Reads | Writes |
|---|---|---|---|
| `01_ky_eda_dataset` | Exploratory analysis of the four core tables | the workbook in `data/raw/` | parquet copies of the sheets in `data/interim/regression/` |
| `02_ky_data_preparation` | Builds the target, simulates the baskets, creates the features and applies the grouped split, which is rebuilt from `data/processed/model_state.csv` with the same code as `03_fm_modelling` | the workbook and `model_state.csv` | `train_df`, `val_df`, `test_df` and `unlisted_df` in `data/processed/regression/`, and `property_split.csv` in `data/interim/regression/` |
| `03_ky_modelling_xgboost_lightgbm` | Fits XGBoost and LightGBM with an 80-candidate random search and chooses the number of trees on validation, without using the test set | `train_df` and `val_df` | the two models (`xgb_log_model.joblib`, `lgbm_log_model.joblib`) and `final_hyperparameters.joblib` in `models/regression/`, and the validation predictions in `data/interim/regression/` |
| `04_ky_ensemble_and_intervals` | Learns the blend weight and the 70% margins on the validation set | the two models and their validation predictions | `ensemble_weight_and_margins.joblib` in `models/regression/` (one dictionary with the weight, the margin of every N and the confidence level) and the validation intervals in `data/interim/regression/` |
| `05_ky_test_evaluation` | Scores the test set once with the frozen weight and margins | the two models, the frozen weight and margins, and `test_df` | the test predictions in `data/interim/regression/` |
| `06_ky_split_and_sampling_audit` | Explains why train, validation and test give different success rates, using bootstrap and random re-split experiments, without tuning anything | the models, the frozen weight and margins, and the three modelling tables | no files, the results are shown in the notebook |
| `07_ky_prediction_table` | Saves the prediction and the 70% interval of every basket row with a train, validation, test or unlisted label | the models, the frozen weight and margins, the modelling tables and `property_split.csv` | `regression_predictions_all_splits.csv` in `data/processed/regression/` |
| `08_ky_reg_prediction_pipeline` | Writes the production pipeline that turns the documented items and property details of one household into an estimate and a 70% interval | the saved models, the frozen weight and margins and the code in `my_val_capstone_01/regression/` | `modeling/reg_predict_pipeline.py` |

## Classification notebooks (`notebooks/`)

The classification team's notebooks `01_fm_data_inventory`, `02_fm_target_definition`, `03_fm_modelling`, `04_fm_inference` and `05_fm_demo` are in `notebooks/` and are not changed by the regression. The regression only reads `data/processed/model_state.csv`, which `02_fm_target_definition` creates, to reproduce the classification split in regression notebook 02.

## Reusable code

The folder `my_val_capstone_01/regression/` holds the Python code of the regression. It is a subpackage of `my_val_capstone_01`, and the front end files of that package are not touched. The files follow the Cookiecutter layout of the template, and each module that creates files has a command line.

| File | Content | Command | Notebook |
|---|---|---|---|
| `config.py` | the folder paths of the regression | none | all |
| `dataset.py` | the household target, the basket simulation and the split of the classification model | `python -m my_val_capstone_01.regression.dataset` | regression 02 |
| `features.py` | the basket features and the five optional regression columns for the classification states, written to `data/processed/classification/model_state_with_reg.csv` | `python -m my_val_capstone_01.regression.features` | regression 02 (the five regression columns are not used by any notebook of this repository) |
| `modeling/train.py` | the search, the choice of the number of trees, the blend weight and the margins | `python -m my_val_capstone_01.regression.modeling.train` | regression 03 and 04 |
| `modeling/predict.py` | `predict_household_value` returns the estimate and its 70% interval for one basket, and `checkpoint_for` returns the basket size that a number of documented items reaches | none | new baskets |

Run the commands from the project root. The modules reproduce the outputs of the notebooks exactly, which is checked by `tests/test_regression.py` (the check of the modelling tables is slow and runs when the environment variable `RUN_SLOW_TESTS` is set to 1). The test set is only scored in regression notebook 05.

## Environment

The random search selects different candidates with different XGBoost versions, so XGBoost is pinned to 3.2.0 in `pyproject.toml`. The saved results were produced with XGBoost 3.2.0 and LightGBM 4.7.0.
