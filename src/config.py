"""Shared configuration values for the project."""

from pathlib import Path


PROJECT_NAME = "Student Performance Decision-Support System"

TARGET_COLUMN = "needs_support"
TARGET_SOURCE_COLUMN = "G3"

PASS_THRESHOLD = 10
POSITIVE_CLASS = 1

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
DEFAULT_CLASSIFICATION_THRESHOLD = 0.50

EARLY_WARNING_EXPERIMENT = "early_warning"
PROGRESS_INFORMED_EXPERIMENT = "progress_informed"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student-por.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_INDICES_PATH = PROCESSED_DATA_DIR / "train_indices.csv"
TEST_INDICES_PATH = PROCESSED_DATA_DIR / "test_indices.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"