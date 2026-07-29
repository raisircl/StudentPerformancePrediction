"""
Streamlit application for the Student Performance Prediction System.

Responsibilities:
- Display the project interface
- Accept student academic information
- Validate user input
- Call the prediction function
- Display predicted final marks
- Show beginner-friendly error messages

This file does not train or evaluate the Machine Learning model.
"""

import sys
from pathlib import Path

import streamlit as st


# Add the project root directory to Python's import path.
# This allows app.py to import modules from the src folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.predict import predict_student_marks  # noqa: E402


def configure_page() -> None:
    """
    Configure the Streamlit browser page.

    This function sets the title, icon, layout, and sidebar state.
    """

    st.set_page_config(
        page_title="Student Performance Prediction",
        page_icon="🎓",
        layout="centered",
        initial_sidebar_state="expanded",
    )


def display_header() -> None:
    """
    Display the application title and introductory information.
    """

    st.title("🎓 Student Performance Prediction System")

    st.write(
        "Enter the student's academic information to predict "
        "the expected final examination marks."
    )

    st.info(
        "This application uses a trained Linear Regression model "
        "to estimate student performance."
    )


def display_sidebar() -> None:
    """
    Display project information in the Streamlit sidebar.
    """

    with st.sidebar:
        st.header("Project Information")

        st.write(
            """
            **Project:** Student Performance Prediction

            **Algorithm:** Linear Regression

            **Framework:** Streamlit

            **Model Library:** Scikit-learn
            """
        )

        st.divider()

        st.subheader("Input Features")

        st.write(
            """
            - Attendance percentage
            - Daily study hours
            - Previous examination marks
            - Assignment marks
            - Internal assessment marks
            """
        )

        st.divider()

        st.caption(
            "The prediction is an estimate generated from historical data."
        )


def get_student_inputs() -> tuple[
    float,
    float,
    float,
    float,
    float,
]:
    """
    Display input controls and collect student information.

    Returns:
        A tuple containing:
        - attendance
        - study hours
        - previous marks
        - assignment marks
        - internal marks
    """

    st.subheader("Student Academic Information")

    left_column, right_column = st.columns(2)

    with left_column:
        attendance = st.number_input(
            label="Attendance Percentage",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0,
            help="Enter the student's attendance percentage.",
        )

        previous_marks = st.number_input(
            label="Previous Examination Marks",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=1.0,
            help="Enter marks obtained in the previous examination.",
        )

        internal_marks = st.number_input(
            label="Internal Assessment Marks",
            min_value=0.0,
            max_value=100.0,
            value=65.0,
            step=1.0,
            help="Enter internal assessment marks.",
        )

    with right_column:
        study_hours = st.number_input(
            label="Daily Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=4.0,
            step=0.5,
            help="Enter average study hours per day.",
        )

        assignment_marks = st.number_input(
            label="Assignment Marks",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0,
            help="Enter marks obtained in assignments.",
        )

    return (
        attendance,
        study_hours,
        previous_marks,
        assignment_marks,
        internal_marks,
    )


def display_input_summary(
    attendance: float,
    study_hours: float,
    previous_marks: float,
    assignment_marks: float,
    internal_marks: float,
) -> None:
    """
    Display the entered student information before prediction.

    Args:
        attendance: Attendance percentage.
        study_hours: Daily study hours.
        previous_marks: Previous examination marks.
        assignment_marks: Assignment marks.
        internal_marks: Internal assessment marks.
    """

    with st.expander("Review entered information"):
        summary_data = {
            "Attendance": f"{attendance:.1f}%",
            "Study Hours": f"{study_hours:.1f} hours",
            "Previous Marks": f"{previous_marks:.1f}",
            "Assignment Marks": f"{assignment_marks:.1f}",
            "Internal Marks": f"{internal_marks:.1f}",
        }

        st.table(summary_data)


def classify_performance(predicted_marks: float) -> str:
    """
    Return a simple performance category based on predicted marks.

    Args:
        predicted_marks: Predicted final examination marks.

    Returns:
        Student performance category.
    """

    if predicted_marks >= 80:
        return "Excellent"

    if predicted_marks >= 65:
        return "Good"

    if predicted_marks >= 50:
        return "Average"

    if predicted_marks >= 33:
        return "Needs Improvement"

    return "At Risk"


def display_prediction_result(predicted_marks: float) -> None:
    """
    Display the prediction result and performance category.

    Args:
        predicted_marks: Predicted final examination marks.
    """

    performance_category = classify_performance(predicted_marks)

    st.success("Prediction completed successfully.")

    result_column, category_column = st.columns(2)

    with result_column:
        st.metric(
            label="Predicted Final Marks",
            value=f"{predicted_marks:.2f}",
        )

    with category_column:
        st.metric(
            label="Performance Category",
            value=performance_category,
        )

    st.progress(
        min(
            max(int(round(predicted_marks)), 0),
            100,
        )
    )

    if predicted_marks < 33:
        st.warning(
            "The predicted marks indicate that the student may require "
            "immediate academic support."
        )
    elif predicted_marks < 50:
        st.warning(
            "The student may benefit from additional practice and guidance."
        )
    else:
        st.info(
            "The prediction should be used as an academic support indicator, "
            "not as a guaranteed examination result."
        )


def main() -> None:
    """
    Run the Student Performance Prediction Streamlit application.
    """

    configure_page()
    display_header()
    display_sidebar()

    with st.form("student_prediction_form"):
        (
            attendance,
            study_hours,
            previous_marks,
            assignment_marks,
            internal_marks,
        ) = get_student_inputs()

        submit_button = st.form_submit_button(
            label="Predict Final Marks",
            type="primary",
            use_container_width=True,
        )

    display_input_summary(
        attendance=attendance,
        study_hours=study_hours,
        previous_marks=previous_marks,
        assignment_marks=assignment_marks,
        internal_marks=internal_marks,
    )

    if submit_button:
        try:
            with st.spinner("Analyzing student performance..."):
                predicted_marks = predict_student_marks(
                    attendance=attendance,
                    study_hours=study_hours,
                    previous_marks=previous_marks,
                    assignment_marks=assignment_marks,
                    internal_marks=internal_marks,
                )

            display_prediction_result(predicted_marks)

        except FileNotFoundError as error:
            st.error(str(error))

            st.warning(
                "Train the Machine Learning model before running predictions."
            )

            st.code(
                "python src/train_model.py",
                language="bash",
            )

        except (TypeError, ValueError) as error:
            st.error(f"Invalid input or model error: {error}")

        except Exception as error:
            st.error(
                "An unexpected error occurred while generating the prediction."
            )

            st.exception(error)


if __name__ == "__main__":
    main()