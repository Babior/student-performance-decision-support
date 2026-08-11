from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_numeric_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )


def build_categorical_pipeline():
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


def build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            ("numeric", build_numeric_pipeline(), numeric_features),
            ("categorical", build_categorical_pipeline(), categorical_features)
        ]
    )