Getting started
===============

Install the dependencies with `uv sync` (Python 3.12). The data folder is not tracked by git,
so the raw workbook `myVal_Synthetic_Datasets_release_v2.xlsx` has to be placed in `data/raw/`
before anything is run.

The regression reads one file of the classification side, so the notebooks are run in this order:

1. `notebooks/02_fm_target_definition.ipynb` creates `data/processed/model_state.csv`.
2. `notebooks/regression/01_ky` to `07_ky` build and evaluate the household-value regression, and `08_ky` writes the production pipeline `modeling/reg_predict_pipeline.py`. Notebook `03_ky` saves the two models and the final hyperparameters, and notebook `04_ky` saves the blend weight and margins, all as joblib files in `models/regression/`. Notebooks `05_ky` to `08_ky` read them, so `04_ky` has to be run first.
3. The classification notebooks `notebooks/01_fm` to `05_fm` are the team's originals. They are run as described in the README and do not need the regression.

The regression code can also be run from the project root with
`python -m my_val_capstone_01.regression.dataset`, and `python -m my_val_capstone_01.regression.modeling.train`. The command
`python -m my_val_capstone_01.regression.features` is optional: it adds five regression columns to the
recommendation states and writes `data/processed/classification/model_state_with_reg.csv`, which no notebook of this repository uses.
See `regression.md` for the details of every notebook and folder.
