# 🎓 Student Performance Prediction System

A complete beginner-friendly Machine Learning project built using **Python**, **Scikit-learn**, **Joblib**, and **Streamlit**.

The application predicts a student's **Final Examination Marks** based on academic performance such as attendance, study hours, previous marks, assignment marks, and internal assessment.

---

# Project Overview

This project demonstrates the complete Machine Learning development lifecycle.

- Problem Understanding
- Dataset Analysis
- Data Preprocessing
- Feature Selection
- Model Training
- Model Evaluation
- Model Serialization
- Prediction
- Streamlit Web Application
- Deployment

The project follows a clean, modular architecture suitable for beginners as well as professional software development.

---

# Project Objective

To predict the expected final examination marks of a student using a trained **Linear Regression** Machine Learning model.

The prediction is based on the following academic features:

- Attendance
- Study Hours
- Previous Marks
- Assignment Marks
- Internal Marks

The model predicts:

- Final Marks

---

# Technology Stack

| Technology | Purpose |
|------------|---------|
| Python 3.13+ | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Computing |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Visualization |
| Scikit-learn | Machine Learning |
| Joblib | Model Serialization |
| Streamlit | Web Application |
| VS Code | Development Environment |
| Git & GitHub | Version Control |

---

# Project Structure

```
StudentPerformancePrediction/
│
├── app/
│   └── app.py
│
├── assets/
│
├── data/
│   └── student_performance.csv
│
├── model/
│   └── linear_regression_model.pkl
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── predict.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Machine Learning Workflow

```
Problem Statement
        │
        ▼
Collect Dataset
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Selection
        │
        ▼
Target Variable
        │
        ▼
Train-Test Split
        │
        ▼
Train Linear Regression Model
        │
        ▼
Model Evaluation
        │
        ▼
Save Model (.pkl)
        │
        ▼
Streamlit Application
        │
        ▼
Prediction
        │
        ▼
Deployment
```

---

# Installation

## Step 1

Clone the repository.

```bash
git clone <repository-url>
```

---

## Step 2

Open the project.

```bash
cd StudentPerformancePrediction
```

---

## Step 3

Create a virtual environment.

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## Step 4

Install required packages.

```bash
pip install -r requirements.txt
```

---

# Dataset

Place the dataset inside:

```
data/
```

Example:

```
data/student_performance.csv
```

Required columns:

| Column Name |
|-------------|
| Attendance |
| StudyHours |
| PreviousMarks |
| AssignmentMarks |
| InternalMarks |
| FinalMarks |

---

# Training the Model

Run:

```bash
python src/train_model.py
```

The trained model will be stored inside:

```
model/
```

Generated file:

```
linear_regression_model.pkl
```

---

# Evaluating the Model

Run:

```bash
python src/evaluate.py
```

The following metrics will be displayed:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

# Running the Streamlit Application

Execute:

```bash
streamlit run app/app.py
```

The application will open in your browser.

---

# Application Features

- User-friendly interface
- Input validation
- Real-time prediction
- Fast model loading
- Modular architecture
- Easy to maintain
- Beginner-friendly code

---

# Project Modules

## config.py

Stores project configuration.

---

## preprocess.py

Loads and preprocesses the dataset.

---

## train_model.py

Trains the Linear Regression model.

---

## evaluate.py

Calculates model performance metrics.

---

## predict.py

Loads the saved model and predicts final marks.

---

## app.py

Provides the Streamlit web interface.

---

# Expected Input

| Feature | Example |
|----------|---------|
| Attendance | 90 |
| StudyHours | 5 |
| PreviousMarks | 76 |
| AssignmentMarks | 82 |
| InternalMarks | 80 |

---

# Expected Output

```
Predicted Final Marks

82.64
```

---

# Learning Outcomes

After completing this project, you will be able to:

- Understand the Machine Learning lifecycle
- Perform Exploratory Data Analysis (EDA)
- Clean and preprocess datasets
- Select features and target variables
- Train a Linear Regression model
- Evaluate model performance
- Save and load Machine Learning models
- Build interactive Streamlit applications
- Deploy Machine Learning projects
- Organize projects using clean architecture

---

# Future Enhancements

- Polynomial Regression
- Random Forest Regression
- XGBoost Regression
- Hyperparameter Tuning
- Feature Scaling
- Cross Validation
- Model Comparison
- Cloud Deployment
- Database Integration
- User Authentication

---

# Developed For

**Machine Learning with Python**

Beginner Project Series

Project 1

Student Performance Prediction System

---

# License

This project is developed for educational purposes.

Students are encouraged to modify, improve, and extend the project for learning and practice.