from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import List


def build_numeric_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )


def build_categorical_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", build_numeric_pipeline(), numeric_features),
            ("categorical", build_categorical_pipeline(), categorical_features)
        ]
    )


def build_model_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    classifier: BaseEstimator
) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )
