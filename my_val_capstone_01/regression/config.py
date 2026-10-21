"""Paths of the household contents-value regression."""

from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # model_state.csv of the classification model
INTERIM_DIR = DATA_DIR / "interim" / "regression"
PROCESSED_DIR = DATA_DIR / "processed" / "regression"
CLASSIFICATION_PROCESSED_DIR = DATA_DIR / "processed" / "classification"

MODELS_DIR = PROJ_ROOT / "models" / "regression"
