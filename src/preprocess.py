"""
Data preprocessing module for the Student Performance Prediction System.

Responsibilities:
- Load the student performance dataset
- Validate required columns
- Remove duplicate records
- Convert required columns to numeric values
- Handle missing values
- Validate acceptable value ranges
- Separate features and target variable

This file does not train or evaluate the Machine Learning model.
"""

from pathlib import Path

import pandas as pd

from config import (
    DATA_FILE_PATH,
    FEATURE_COLUMNS,
    MAX_ATTENDANCE,
    MAX_MARKS,
    MAX_STUDY_HOURS,
    MIN_ATTENDANCE,
    MIN_MARKS,
    MIN_STUDY_HOURS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
)


def load_dataset(file_path: Path = DATA_FILE_PATH) -> pd.DataFrame:
    """
    Load the student performance dataset from a CSV file.

    Args:
        file_path: Path of the CSV dataset.

    Returns:
        A Pandas DataFrame containing the dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset is empty or cannot be read correctly.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file was not found at: {file_path}"
        )

    try:
        dataframe = pd.read_csv(file_path)
    except pd.errors.EmptyDataError as error:
        raise ValueError("The dataset file is empty.") from error
    except pd.errors.ParserError as error:
        raise ValueError(
            "The dataset could not be parsed. Check the CSV format."
        ) from error
    except Exception as error:
        raise ValueError(
            f"An error occurred while reading the dataset: {error}"
        ) from error

    if dataframe.empty:
        raise ValueError("The dataset does not contain any records.")

    return dataframe


def validate_required_columns(dataframe: pd.DataFrame) -> None:
    """
    Validate that all required columns exist in the dataset.

    Args:
        dataframe: Dataset to validate.

    Raises:
        ValueError: If one or more required columns are missing.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        missing_column_names = ", ".join(missing_columns)

        raise ValueError(
            "The dataset is missing the following required columns: "
            f"{missing_column_names}"
        )


def remove_duplicate_records(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.

    Args:
        dataframe: Dataset containing possible duplicate records.

    Returns:
        A new DataFrame without duplicate rows.
    """

    return dataframe.drop_duplicates().copy()


def convert_columns_to_numeric(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert all feature and target columns to numeric values.

    Invalid values are converted to NaN so they can be handled later.

    Args:
        dataframe: Dataset containing student records.

    Returns:
        DataFrame with numeric feature and target columns.
    """

    cleaned_dataframe = dataframe.copy()

    for column in REQUIRED_COLUMNS:
        cleaned_dataframe[column] = pd.to_numeric(
            cleaned_dataframe[column],
            errors="coerce",
        )

    return cleaned_dataframe


def handle_missing_values(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Handle missing values in required columns.

    Missing values are replaced with the median value of their
    respective column.

    Args:
        dataframe: Dataset containing missing values.

    Returns:
        DataFrame with missing values handled.

    Raises:
        ValueError: If a required column contains no valid numeric values.
    """

    cleaned_dataframe = dataframe.copy()

    for column in REQUIRED_COLUMNS:
        if cleaned_dataframe[column].isna().all():
            raise ValueError(
                f"Column '{column}' does not contain any valid numeric values."
            )

        median_value = cleaned_dataframe[column].median()

        cleaned_dataframe[column] = cleaned_dataframe[column].fillna(
            median_value
        )

    return cleaned_dataframe


def validate_data_ranges(dataframe: pd.DataFrame) -> None:
    """
    Validate the acceptable range of student data.

    Expected ranges:
    - Attendance: 0 to 100
    - StudyHours: 0 to 24
    - Marks columns: 0 to 100

    Args:
        dataframe: Cleaned student dataset.

    Raises:
        ValueError: If any value is outside the allowed range.
    """

    attendance_is_invalid = ~dataframe["Attendance"].between(
        MIN_ATTENDANCE,
        MAX_ATTENDANCE,
    )

    if attendance_is_invalid.any():
        raise ValueError(
            "Attendance values must be between "
            f"{MIN_ATTENDANCE} and {MAX_ATTENDANCE}."
        )

    study_hours_are_invalid = ~dataframe["StudyHours"].between(
        MIN_STUDY_HOURS,
        MAX_STUDY_HOURS,
    )

    if study_hours_are_invalid.any():
        raise ValueError(
            "StudyHours values must be between "
            f"{MIN_STUDY_HOURS} and {MAX_STUDY_HOURS}."
        )

    marks_columns = [
        "PreviousMarks",
        "AssignmentMarks",
        "InternalMarks",
        TARGET_COLUMN,
    ]

    for column in marks_columns:
        marks_are_invalid = ~dataframe[column].between(
            MIN_MARKS,
            MAX_MARKS,
        )

        if marks_are_invalid.any():
            raise ValueError(
                f"{column} values must be between "
                f"{MIN_MARKS} and {MAX_MARKS}."
            )


def preprocess_dataset(
    file_path: Path = DATA_FILE_PATH,
) -> pd.DataFrame:
    """
    Execute the complete dataset preprocessing workflow.

    Steps:
    1. Load the CSV dataset
    2. Validate required columns
    3. Select only required columns
    4. Remove duplicate records
    5. Convert values to numeric
    6. Handle missing values
    7. Validate data ranges

    Args:
        file_path: Path of the CSV dataset.

    Returns:
        A clean DataFrame ready for model training.
    """

    dataframe = load_dataset(file_path)

    validate_required_columns(dataframe)

    dataframe = dataframe[REQUIRED_COLUMNS].copy()

    dataframe = remove_duplicate_records(dataframe)

    dataframe = convert_columns_to_numeric(dataframe)

    dataframe = handle_missing_values(dataframe)

    validate_data_ranges(dataframe)

    dataframe.reset_index(drop=True, inplace=True)

    return dataframe


def prepare_features_and_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate input features and target values.

    Args:
        dataframe: Preprocessed student performance dataset.

    Returns:
        A tuple containing:
        - X: Input feature DataFrame
        - y: Target Series containing FinalMarks
    """

    features = dataframe[FEATURE_COLUMNS].copy()
    target = dataframe[TARGET_COLUMN].copy()

    return features, target


def get_preprocessed_data(
    file_path: Path = DATA_FILE_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load, clean, and separate the dataset.

    This is the main function that other project modules should call.

    Args:
        file_path: Path of the CSV dataset.

    Returns:
        A tuple containing input features and target values.
    """

    dataframe = preprocess_dataset(file_path)

    features, target = prepare_features_and_target(dataframe)

    return features, target


if __name__ == "__main__":
    try:
        clean_data = preprocess_dataset()

        print("Dataset preprocessing completed successfully.")
        print(f"Total records: {len(clean_data)}")
        print(f"Total columns: {len(clean_data.columns)}")
        print("\nFirst five records:")
        print(clean_data.head())

    except (FileNotFoundError, ValueError) as error:
        print(f"Preprocessing failed: {error}")