"""
Utility functions and input mappings for the Student Performance
Decision-Support System.
"""

import json
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


def load_final_model():
    """Load the final trained machine-learning pipeline."""
    return joblib.load(MODEL_PATH)


def load_model_metadata():
    """Load metadata associated with the final trained model."""
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


EDUCATION_OPTIONS = {
    "No formal education": 0,
    "Primary education": 1,
    "5th to 9th grade": 2,
    "Secondary education": 3,
    "Higher education": 4,
}


TRAVEL_TIME_OPTIONS = {
    "Less than 15 minutes": 1,
    "15 to 30 minutes": 2,
    "30 minutes to 1 hour": 3,
    "More than 1 hour": 4,
}


STUDY_TIME_OPTIONS = {
    "Less than 2 hours": 1,
    "2 to 5 hours": 2,
    "5 to 10 hours": 3,
    "More than 10 hours": 4,
}


FIVE_POINT_OPTIONS = {
    "Very low": 1,
    "Low": 2,
    "Moderate": 3,
    "High": 4,
    "Very high": 5,
}


RELATIONSHIP_OPTIONS = {
    "Very poor": 1,
    "Poor": 2,
    "Average": 3,
    "Good": 4,
    "Excellent": 5,
}


HEALTH_OPTIONS = {
    "Very poor": 1,
    "Poor": 2,
    "Average": 3,
    "Good": 4,
    "Very good": 5,
}


YES_NO_OPTIONS = {
    "No": "no",
    "Yes": "yes",
}


SEX_OPTIONS = {
    "Female": "F",
    "Male": "M",
}


ADDRESS_OPTIONS = {
    "Urban": "U",
    "Rural": "R",
}


FAMILY_SIZE_OPTIONS = {
    "3 or fewer": "LE3",
    "More than 3": "GT3",
}


PARENT_STATUS_OPTIONS = {
    "Living together": "T",
    "Apart": "A",
}


JOB_OPTIONS = {
    "At home": "at_home",
    "Health": "health",
    "Other": "other",
    "Services": "services",
    "Teacher": "teacher",
}


REASON_OPTIONS = {
    "Course preference": "course",
    "Close to home": "home",
    "School reputation": "reputation",
    "Other": "other",
}


GUARDIAN_OPTIONS = {
    "Mother": "mother",
    "Father": "father",
    "Other": "other",
}


def build_student_dataframe(student_data):
    """Build a one-row DataFrame for final-model inference."""

    expected_columns = [
        "school",
        "sex",
        "age",
        "address",
        "famsize",
        "Pstatus",
        "Medu",
        "Fedu",
        "Mjob",
        "Fjob",
        "reason",
        "guardian",
        "traveltime",
        "studytime",
        "failures",
        "schoolsup",
        "famsup",
        "paid",
        "activities",
        "nursery",
        "higher",
        "internet",
        "romantic",
        "famrel",
        "freetime",
        "goout",
        "Dalc",
        "Walc",
        "health",
        "absences",
        "G1",
        "G2",
    ]

    return pd.DataFrame(
        [[student_data[column] for column in expected_columns]],
        columns=expected_columns,
    )