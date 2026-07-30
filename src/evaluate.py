"""
Model evaluation module for the Student Performance Prediction System.

Responsibilities:
- Load and preprocess the dataset
- Split the dataset using the same configuration as training
- Load the saved Linear Regression model
- Generate predictions for the testing dataset
- Calculate evaluation metrics
- Display actual and predicted marks

This file does not train or save the model.
"""

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

try:
    from src.config import MODEL_FILE_PATH
except ModuleNotFoundError:
    from config import MODEL_FILE_PATH

try:
    from src.preprocess import get_preprocessed_data
except ModuleNotFoundError:
    from preprocess import get_preprocessed_data

try:
    from src.train_model import split_dataset
except ModuleNotFoundError:
    from train_model import split_dataset


@dataclass(frozen=True)
class EvaluationResult:
    """
    Store the calculated model evaluation metrics.

    Attributes:
        mean_absolute_error: Average absolute prediction error.
        mean_squared_error: Average squared prediction error.
        root_mean_squared_error: Square root of MSE.
        r2_score: Percentage of target variation explained by the model.
    """

    mean_absolute_error: float
    mean_squared_error: float
    root_mean_squared_error: float
    r2_score: float


def load_model() -> LinearRegression:
    """
    Load the trained Linear Regression model from disk.

    Returns:
        The trained Linear Regression model.

    Raises:
        FileNotFoundError: If the trained model file does not exist.
        ValueError: If the file cannot be loaded or contains an invalid model.
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


def generate_predictions(
    model: LinearRegression,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Generate predictions for the testing dataset.

    Args:
        model: Trained Linear Regression model.
        X_test: Testing feature data.

    Returns:
        NumPy array containing predicted final marks.
    """

    predictions = model.predict(X_test)

    return np.asarray(predictions, dtype=float)


def calculate_metrics(
    y_test: pd.Series,
    predictions: np.ndarray,
) -> EvaluationResult:
    """
    Calculate regression evaluation metrics.

    Metrics:
    - MAE
    - MSE
    - RMSE
    - R² score

    Args:
        y_test: Actual final marks from the testing dataset.
        predictions: Marks predicted by the model.

    Returns:
        EvaluationResult containing all calculated metrics.
    """

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    return EvaluationResult(
        mean_absolute_error=float(mae),
        mean_squared_error=float(mse),
        root_mean_squared_error=float(rmse),
        r2_score=float(r2),
    )


def create_comparison_table(
    y_test: pd.Series,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Create a table comparing actual and predicted marks.

    Args:
        y_test: Actual final marks.
        predictions: Predicted final marks.

    Returns:
        DataFrame containing actual marks, predicted marks, and error.
    """

    actual_values = y_test.reset_index(drop=True)

    comparison = pd.DataFrame(
        {
            "ActualMarks": actual_values,
            "PredictedMarks": np.round(predictions, 2),
        }
    )

    comparison["AbsoluteError"] = np.round(
        np.abs(
            comparison["ActualMarks"]
            - comparison["PredictedMarks"]
        ),
        2,
    )

    return comparison


def display_evaluation_results(
    result: EvaluationResult,
) -> None:
    """
    Display the calculated model evaluation metrics.

    Args:
        result: Object containing evaluation metrics.
    """

    print("\nModel Evaluation Results")
    print("-" * 50)
    print(
        f"Mean Absolute Error (MAE)  : "
        f"{result.mean_absolute_error:.4f}"
    )
    print(
        f"Mean Squared Error (MSE)   : "
        f"{result.mean_squared_error:.4f}"
    )
    print(
        f"Root Mean Squared Error    : "
        f"{result.root_mean_squared_error:.4f}"
    )
    print(
        f"R² Score                   : "
        f"{result.r2_score:.4f}"
    )


def interpret_r2_score(score: float) -> str:
    """
    Return a simple interpretation of the R² score.

    This interpretation is intended for classroom explanation and should
    not be treated as a universal industry threshold.

    Args:
        score: Calculated R² score.

    Returns:
        Human-readable model performance interpretation.
    """

    if score >= 0.90:
        return "Excellent model performance."

    if score >= 0.75:
        return "Good model performance."

    if score >= 0.50:
        return "Moderate model performance."

    if score >= 0.0:
        return "Weak model performance. Improvement is required."

    return (
        "Poor model performance. The model performs worse than "
        "predicting the average target value."
    )


def execute_evaluation_pipeline() -> tuple[
    EvaluationResult,
    pd.DataFrame,
]:
    """
    Execute the complete model evaluation workflow.

    Steps:
    1. Load and preprocess the dataset
    2. Split it using the same settings as training
    3. Load the saved model
    4. Generate test predictions
    5. Calculate evaluation metrics
    6. Build an actual-versus-predicted comparison table

    Returns:
        A tuple containing:
        - EvaluationResult
        - Actual-versus-predicted comparison DataFrame
    """

    features, target = get_preprocessed_data()

    _, X_test, _, y_test = split_dataset(
        features,
        target,
    )

    model = load_model()

    predictions = generate_predictions(
        model,
        X_test,
    )

    evaluation_result = calculate_metrics(
        y_test,
        predictions,
    )

    comparison_table = create_comparison_table(
        y_test,
        predictions,
    )

    return evaluation_result, comparison_table


if __name__ == "__main__":
    try:
        results, comparison = execute_evaluation_pipeline()

        display_evaluation_results(results)

        print(
            "\nPerformance Interpretation: "
            f"{interpret_r2_score(results.r2_score)}"
        )

        print("\nActual vs Predicted Marks")
        print("-" * 50)
        print(comparison.to_string(index=False))

        print("\nModel evaluation completed successfully.")

    except (
        FileNotFoundError,
        ValueError,
        OSError,
    ) as error:
        print(f"Model evaluation failed: {error}")