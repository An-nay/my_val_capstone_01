"""The grouped train / validation / test split of notebook 03_fm_modelling, shared by the build scripts.

GroupShuffleSplit by Property_ID, seed 42, 70% train then the rest split 50/50, so a household is
never in two partitions. This reproduces the split the saved models were trained with.
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_properties(model_state: pd.DataFrame) -> dict:
    """Return {"train": set, "validation": set, "test": set} of Property_IDs."""
    groups = model_state["Property_ID"]
    train_idx, temp_idx = next(GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=42).split(model_state, groups=groups))
    temp = model_state.iloc[temp_idx]
    val_idx, test_idx = next(GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=42).split(temp, groups=temp["Property_ID"]))
    return {
        "train": set(model_state.iloc[train_idx]["Property_ID"]),
        "validation": set(temp.iloc[val_idx]["Property_ID"]),
        "test": set(temp.iloc[test_idx]["Property_ID"]),
    }

