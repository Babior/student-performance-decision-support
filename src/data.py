"""Shared functions for loading and preparing the student dataset."""

from pathlib import Path

import pandas as pd

from src.config import (
    PASS_THRESHOLD,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_INDICES_PATH,
    TRAIN_INDICES_PATH,
)


REQUIRED_COLUMNS = {"G1", "G2", "G3"}


def load_student_data(file_path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the Portuguese student-performance dataset.

    Args:
        file_path: Path to the semicolon-separated CSV file.

    Returns:
        A pandas DataFrame containing the student data.

    Raises:
        FileNotFoundError: If the dataset does not exist.
        ValueError: If the dataset cannot be loaded correctly.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    try:
        dataframe = pd.read_csv(path, sep=";")
    except Exception as error:
        raise ValueError(f"Unable to load dataset: {error}") from error

    return dataframe


def validate_required_columns(dataframe: pd.DataFrame) -> None:
    """Confirm that the required grade columns exist.

    Args:
        dataframe: Student-performance DataFrame.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )


def create_target(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create the binary academic-support target.

    Target values:
        1: May require academic support.
        0: Likely to pass.

    Args:
        dataframe: Original student-performance DataFrame.

    Returns:
        A copy of the DataFrame with the target column added.
    """
    validate_required_columns(dataframe)

    result = dataframe.copy()
    result[TARGET_COLUMN] = (
        result["G3"] < PASS_THRESHOLD
    ).astype(int)

    return result


def get_feature_sets(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Create the shared feature sets and target.

    Returns:
        X_base:
            All approved features except G3 and the target.
        X_early:
            Early-warning features excluding G1 and G2.
        X_progress:
            Progress-informed features including G1 and G2.
        y:
            Binary academic-support target.
    """
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' has not been created."
        )

    X_base = dataframe.drop(columns=["G3", TARGET_COLUMN])
    X_early = X_base.drop(columns=["G1", "G2"])
    X_progress = X_base.copy()
    y = dataframe[TARGET_COLUMN].copy()

    return X_base, X_early, X_progress, y


def save_split_indices(
    train_indices,
    test_indices,
) -> None:
    """Save the shared training and testing row indices."""

    TRAIN_INDICES_PATH.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"index": train_indices}).to_csv(
        TRAIN_INDICES_PATH,
        index=False,
    )

    pd.DataFrame({"index": test_indices}).to_csv(
        TEST_INDICES_PATH,
        index=False,
    )


def load_split_indices() -> tuple[list[int], list[int]]:
    """Load the saved shared training and testing row indices.

    Raises:
        FileNotFoundError: If either split-index file is missing.
    """
    if not TRAIN_INDICES_PATH.exists():
        raise FileNotFoundError(
            f"Training indices not found: {TRAIN_INDICES_PATH}"
        )

    if not TEST_INDICES_PATH.exists():
        raise FileNotFoundError(
            f"Testing indices not found: {TEST_INDICES_PATH}"
        )

    train_indices = (
        pd.read_csv(TRAIN_INDICES_PATH)["index"].astype(int).tolist()
    )

    test_indices = (
        pd.read_csv(TEST_INDICES_PATH)["index"].astype(int).tolist()
    )

    return train_indices, test_indices