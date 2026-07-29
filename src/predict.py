"""
Prediction module for the Student Performance Prediction System.

Responsibilities:
- Validate user input
- Load the saved Linear Regression model
- Convert student input into the correct feature format
- Predict final examination marks
- Return a clean numeric prediction

This file does not train or evaluate the model.
"""

from functools import lru_cache

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import (
    FEATURE_COLUMNS,
    MAX_ATTENDANCE,
    MAX_MARKS,
    MAX_STUDY_HOURS,
    MIN_ATTENDANCE,
    MIN_MARKS,
    MIN_STUDY_HOURS,
    MODEL_FILE_PATH,
)


@lru_cache(maxsize=1)
def load_trained_model() -> LinearRegression:
    """
    Load the trained Linear Regression model from disk.

    The model is cached after the first load so that repeated predictions
    do not read the model file from disk every time.

    Returns:
        The trained Linear Regression model.

    Raises:
        FileNotFoundError: If the saved model file does not exist.
        ValueError: If the model cannot be loaded or is not valid.
    """

    if not MODEL_FILE_PATH.exists():
        raise FileNotFoundError(
            "The trained model file was not found at: "
            f"{MODEL_FILE_PATH}. Run src/train_model.py first."
        )

    try:
        model = joblib.load(MODEL_FILE_PATH)
    except Exception as error:
        raise ValueError(
            f"Unable to load the trained model: {error}"
        ) from error

    if not isinstance(model, LinearRegression):
        raise ValueError(
            "The saved file does not contain a valid "
            "LinearRegression model."
        )

    return model


def validate_numeric_value(
    value: float,
    field_name: str,
    minimum_value: float,
    maximum_value: float,
) -> float:
    """
    Validate that a value is numeric and falls within an allowed range.

    Args:
        value: Value entered by the user.
        field_name: Name of the field being validated.
        minimum_value: Minimum acceptable value.
        maximum_value: Maximum acceptable value.

    Returns:
        The validated value as a float.

    Raises:
        TypeError: If the value is not numeric.
        ValueError: If the value is outside the allowed range.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{field_name} must be a numeric value."
        )

    numeric_value = float(value)

    if not minimum_value <= numeric_value <= maximum_value:
        raise ValueError(
            f"{field_name} must be between "
            f"{minimum_value} and {maximum_value}."
        )

    return numeric_value


def validate_student_inputs(
    attendance: float,
    study_hours: float,
    previous_marks: float,
    assignment_marks: float,
    internal_marks: float,
) -> dict[str, float]:
    """
    Validate all student input values.

    Args:
        attendance: Student attendance percentage.
        study_hours: Average study hours per day.
        previous_marks: Marks obtained in the previous examination.
        assignment_marks: Assignment marks.
        internal_marks: Internal assessment marks.

    Returns:
        Dictionary containing validated student information.

    Raises:
        TypeError: If any value is not numeric.
        ValueError: If any value is outside its allowed range.
    """

    return {
        "Attendance": validate_numeric_value(
            attendance,
            "Attendance",
            MIN_ATTENDANCE,
            MAX_ATTENDANCE,
        ),
        "StudyHours": validate_numeric_value(
            study_hours,
            "Study Hours",
            MIN_STUDY_HOURS,
            MAX_STUDY_HOURS,
        ),
        "PreviousMarks": validate_numeric_value(
            previous_marks,
            "Previous Marks",
            MIN_MARKS,
            MAX_MARKS,
        ),
        "AssignmentMarks": validate_numeric_value(
            assignment_marks,
            "Assignment Marks",
            MIN_MARKS,
            MAX_MARKS,
        ),
        "InternalMarks": validate_numeric_value(
            internal_marks,
            "Internal Marks",
            MIN_MARKS,
            MAX_MARKS,
        ),
    }


def create_prediction_dataframe(
    student_data: dict[str, float],
) -> pd.DataFrame:
    """
    Convert validated student data into a one-row DataFrame.

    The column order must match the feature order used during training.

    Args:
        student_data: Dictionary containing validated feature values.

    Returns:
        A one-row Pandas DataFrame ready for model prediction.
    """

    return pd.DataFrame(
        [[student_data[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )


def limit_prediction_to_marks_range(
    predicted_marks: float,
) -> float:
    """
    Limit the prediction to the valid marks range of 0 to 100.

    Linear Regression can theoretically predict values below 0 or
    above 100. This function keeps the displayed result within the
    valid examination marks range.

    Args:
        predicted_marks: Raw prediction returned by the model.

    Returns:
        Prediction limited to the valid marks range.
    """

    return max(
        MIN_MARKS,
        min(float(predicted_marks), MAX_MARKS),
    )


def predict_student_marks(
    attendance: float,
    study_hours: float,
    previous_marks: float,
    assignment_marks: float,
    internal_marks: float,
) -> float:
    """
    Predict a student's final examination marks.

    Steps:
    1. Validate user input
    2. Create a one-row feature DataFrame
    3. Load the trained model
    4. Generate the prediction
    5. Limit the result to 0-100
    6. Round the result to two decimal places

    Args:
        attendance: Student attendance percentage.
        study_hours: Average study hours per day.
        previous_marks: Previous examination marks.
        assignment_marks: Assignment marks.
        internal_marks: Internal assessment marks.

    Returns:
        Predicted final marks rounded to two decimal places.
    """

    student_data = validate_student_inputs(
        attendance=attendance,
        study_hours=study_hours,
        previous_marks=previous_marks,
        assignment_marks=assignment_marks,
        internal_marks=internal_marks,
    )

    prediction_input = create_prediction_dataframe(student_data)

    model = load_trained_model()

    predicted_value = model.predict(prediction_input)[0]

    final_prediction = limit_prediction_to_marks_range(
        predicted_value
    )

    return round(final_prediction, 2)


if __name__ == "__main__":
    try:
        predicted_marks = predict_student_marks(
            attendance=90,
            study_hours=5,
            previous_marks=76,
            assignment_marks=82,
            internal_marks=80,
        )

        print("Student Performance Prediction")
        print("-" * 40)
        print(f"Predicted Final Marks: {predicted_marks}")

    except (
        FileNotFoundError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Prediction failed: {error}")