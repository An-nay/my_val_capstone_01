"""Household contents-value regression: modelling tables.

Builds the target (total non-vehicle contents value of a home), the simulated baskets of
N = 10, 20, 30 and 50 documented items, and the train / validation / test / unlisted tables
(notebook 02_ky_data_preparation). The split follows the grouped split of the classification
model (03_fm_modelling), rebuilt from data/processed/model_state.csv.
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import typer

from my_val_capstone_01.regression.config import INTERIM_DIR, PROCESSED_DATA_DIR, PROCESSED_DIR, RAW_DATA_DIR
from my_val_capstone_01.regression.features import featurize_sample

app = typer.Typer()

SHEETS = ["assets", "properties", "customers", "customer_profiles"]
SAMPLE_SIZES = [10, 20, 30, 50]
REPEATS_PER_N = 5
MIN_ITEMS_REQUIRED = 10
RANDOM_STATE = 42


def load_sheets(xlsx_path: Path, cache_dir: Path) -> dict:
    """Read the four sheets, caching each of them as parquet because the workbook is slow."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    sheets = {}
    for name in SHEETS:
        cached = cache_dir / f"{name}.parquet"
        if not cached.exists():
            pd.read_excel(xlsx_path, sheet_name=name).to_parquet(cached, index=False)
        sheets[name] = pd.read_parquet(cached)
    return sheets


def build_household_table(sheets: dict) -> tuple:
    """Return the valued non-vehicle items, the household table with its target, the categories."""
    assets, properties = sheets["assets"], sheets["properties"]
    contents_categories = sorted(assets.loc[assets["Category"] != "Vehicles", "Category"].unique())
    assets_contents = assets[assets["Category"] != "Vehicles"].copy()
    # items without an estimated value would count as 0 in the target, so they are left out
    assets_contents = assets_contents[assets_contents["Current_Estimated_Value"].notna()].copy()
    true_value = (
        assets_contents.groupby("Property_ID")["Current_Estimated_Value"]
        .sum()
        .rename("true_total_value")
        .reset_index()
    )
    customers_features = sheets["customers"][
        ["Customer_ID", "Age_Band", "Membership_Level"]
    ].copy()
    profile_features = sheets["customer_profiles"][
        ["Customer_ID", "Income_Band", "Household_Type"]
    ].copy()
    prop_features = properties[
        [
            "Property_ID",
            "Customer_ID",
            "Property_Type",
            "Occupancy_Type",
            "Number_Of_Occupants",
            "Number_Of_Bedrooms",
            "Property_Valuation_AUD",
        ]
    ].copy()
    prop_features = (
        prop_features.merge(customers_features, on="Customer_ID", how="left")
        .merge(profile_features, on="Customer_ID", how="left")
        .fillna(
            {
                "Age_Band": "Unknown",
                "Membership_Level": "Unknown",
                "Income_Band": "Unknown",
                "Household_Type": "Unknown",
            }
        )
    )
    household = prop_features.merge(true_value, on="Property_ID", how="inner")
    return assets_contents, household, contents_categories


def generate_samples(
    items_df: pd.DataFrame, rng, sample_sizes=SAMPLE_SIZES, repeats=REPEATS_PER_N
):
    """Draw N items without replacement from one home, ``repeats`` times for each N."""
    n_available = len(items_df)
    if n_available < MIN_ITEMS_REQUIRED:
        return
    for n in sample_sizes:
        if n > n_available:
            continue
        for _ in range(repeats):
            idx = rng.choice(n_available, size=n, replace=False)
            yield n, items_df.iloc[idx]


def simulate_baskets(
    assets_contents: pd.DataFrame,
    household: pd.DataFrame,
    contents_categories: list,
    properties: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Simulate the documented baskets of every home and turn each one into a feature row.

    Notebook 02_ky_data_preparation first draws a worked example (two baskets of 10 and two of
    20 items from the first home with at least 20 items) from the same random generator. Those
    draws are repeated here, so that the tables are identical to the ones of the notebook.
    """
    rng = np.random.default_rng(random_state)
    assets_by_property = {
        pid: grp.reset_index(drop=True) for pid, grp in assets_contents.groupby("Property_ID")
    }
    items_per_prop = (
        pd.Series({p: len(g) for p, g in assets_by_property.items()})
        .reindex(properties["Property_ID"])
        .fillna(0)
        .astype(int)
    )
    example_pid = items_per_prop[items_per_prop >= 20].index[0]
    for _ in generate_samples(assets_by_property[example_pid], rng, [10, 20], 2):
        pass

    household_indexed = household.set_index("Property_ID")
    rows = []
    for pid, items_df in assets_by_property.items():
        if pid not in household_indexed.index:
            continue
        prop_row = household_indexed.loc[pid]
        for n, sample_df in generate_samples(items_df, rng):
            feats = featurize_sample(sample_df, prop_row, contents_categories)
            feats["Property_ID"] = pid
            feats["n_documented"] = n
            feats["true_total_value"] = prop_row["true_total_value"]
            rows.append(feats)
    dataset = pd.DataFrame(rows)
    return pd.get_dummies(dataset, columns=["property_type", "occupancy_type"], drop_first=False)


def rebuild_state_split(model_state: pd.DataFrame) -> pd.Series:
    """Rebuild the classification model's train / validation / test label of every state.

    The two GroupShuffleSplit steps of 03_fm_modelling are repeated: 70% train, then the
    remaining 30% are cut into equal validation and test parts, always grouped by Property_ID
    and with random_state = 42.
    """
    state_pid = model_state["Property_ID"]
    train_idx, temp_idx = next(
        GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=42).split(
            model_state, groups=state_pid
        )
    )
    val_pos, test_pos = next(
        GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=42).split(
            model_state.iloc[temp_idx], groups=state_pid.iloc[temp_idx]
        )
    )
    state_split = pd.Series("", index=model_state.index)
    state_split.iloc[train_idx] = "train"
    state_split.iloc[temp_idx[val_pos]] = "validation"
    state_split.iloc[temp_idx[test_pos]] = "test"
    assert state_split.ne("").all()
    return state_split


def split_by_classification_model(
    dataset: pd.DataFrame,
    properties: pd.DataFrame,
    model_state: pd.DataFrame,
    include_unlisted_in_train: bool = False,
) -> tuple:
    """Give every regression home the label the classification model gave it.

    Homes that never form a classification state have no label to follow. They are kept out
    of train, validation and test and returned as ``unlisted``.
    """
    state_split = rebuild_state_split(model_state)
    property_labels = state_split.groupby(model_state["Property_ID"]).first()
    classifier = property_labels.rename("classifier_split").reset_index()
    prop_split = (
        dataset[["Property_ID"]]
        .drop_duplicates()
        .merge(classifier, on="Property_ID", how="left")
        .fillna({"classifier_split": "not_in_classifier"})
    )
    unlisted_label = "train" if include_unlisted_in_train else "unlisted"
    prop_split["regression_split"] = prop_split["classifier_split"].replace(
        {"not_in_classifier": unlisted_label}
    )
    label = dataset["Property_ID"].map(prop_split.set_index("Property_ID")["regression_split"])
    tables = {
        "train": dataset[label == "train"].reset_index(drop=True),
        "validation": dataset[label == "validation"].reset_index(drop=True),
        "test": dataset[label == "test"].reset_index(drop=True),
        "unlisted": dataset[label == "unlisted"].reset_index(drop=True),
    }
    prop_split = (
        prop_split.merge(properties[["Property_ID", "Customer_ID"]], on="Property_ID", how="left")
        .sort_values(["regression_split", "classifier_split", "Property_ID"])
        .reset_index(drop=True)
    )
    return tables, prop_split


@app.command()
def main(
    xlsx_path: Path = RAW_DATA_DIR / "myVal_Synthetic_Datasets_release_v2.xlsx",
    model_state_path: Path = PROCESSED_DATA_DIR / "model_state.csv",
    interim_dir: Path = INTERIM_DIR,
    processed_dir: Path = PROCESSED_DIR,
):
    """Build train_df, val_df, test_df and unlisted_df and the property split."""
    logger.info("Building the modelling tables of the regression...")
    sheets = load_sheets(xlsx_path, interim_dir)
    assets_contents, household, categories = build_household_table(sheets)
    dataset = simulate_baskets(assets_contents, household, categories, sheets["properties"])
    model_state = pd.read_csv(model_state_path, usecols=["Property_ID", "Next_Category"])
    tables, prop_split = split_by_classification_model(dataset, sheets["properties"], model_state)

    processed_dir.mkdir(parents=True, exist_ok=True)
    prop_split.to_csv(interim_dir / "property_split.csv", index=False)
    file_names = {"train": "train_df", "validation": "val_df", "test": "test_df"}
    for split, name in file_names.items():
        tables[split].to_parquet(processed_dir / f"{name}.parquet", index=False)
    if len(tables["unlisted"]):
        tables["unlisted"].to_parquet(processed_dir / "unlisted_df.parquet", index=False)
    counts = {split: len(table) for split, table in tables.items()}
    logger.success(f"Modelling tables saved (rows per split: {counts}).")


if __name__ == "__main__":
    app()
