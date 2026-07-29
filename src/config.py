"""
Central configuration for the Student Performance Prediction project.

This module contains:
- Project directory paths
- Dataset and model file paths
- Feature and target column names
- Model training configuration

No data preprocessing, model training, or prediction logic should be
written in this file.
"""

from pathlib import Path
from typing import Final


# =========================================================
# PROJECT DIRECTORIES
# =========================================================

# Absolute path of the main project directory.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

# Folder containing the CSV dataset.
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"

# Folder where the trained Machine Learning model will be stored.
MODEL_DIR: Final[Path] = PROJECT_ROOT / "model"

# Folder containing the Streamlit application.
APP_DIR: Final[Path] = PROJECT_ROOT / "app"

# Folder containing optional images and other static resources.
ASSETS_DIR: Final[Path] = PROJECT_ROOT / "assets"


# =========================================================
# FILE PATHS
# =========================================================

# Complete path of the student performance dataset.
DATA_FILE_PATH: Final[Path] = DATA_DIR / "student_performance.csv"

# Complete path where the trained Linear Regression model will be saved.
MODEL_FILE_PATH: Final[Path] = (
    MODEL_DIR / "linear_regression_model.pkl"
)


# =========================================================
# DATASET CONFIGURATION
# =========================================================

# Input columns used by the model to predict student performance.
FEATURE_COLUMNS: Final[list[str]] = [
    "Attendance",
    "StudyHours",
    "PreviousMarks",
    "AssignmentMarks",
    "InternalMarks",
]

# Output column that the Machine Learning model will predict.
TARGET_COLUMN: Final[str] = "FinalMarks"

# All columns required in the dataset.
REQUIRED_COLUMNS: Final[list[str]] = [
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
]


# =========================================================
# MODEL TRAINING CONFIGURATION
# =========================================================

# 20% of the dataset will be used for model testing.
TEST_SIZE: Final[float] = 0.20

# Ensures that the same train-test split is generated every time.
RANDOM_STATE: Final[int] = 42


# =========================================================
# VALID INPUT RANGES
# =========================================================

# These values will be used later for input validation.
MIN_ATTENDANCE: Final[float] = 0.0
MAX_ATTENDANCE: Final[float] = 100.0

MIN_STUDY_HOURS: Final[float] = 0.0
MAX_STUDY_HOURS: Final[float] = 24.0

MIN_MARKS: Final[float] = 0.0
MAX_MARKS: Final[float] = 100.0


def create_required_directories() -> None:
    """
    Create the required project directories if they do not already exist.

    This function is useful when the project is executed for the first
    time and the model or assets folders have not yet been created.
    """

    directories = [
        DATA_DIR,
        MODEL_DIR,
        APP_DIR,
        ASSETS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)