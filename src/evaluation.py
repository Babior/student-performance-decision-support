import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import (
    RANDOM_STATE,
    CV_FOLDS,
    DEFAULT_CLASSIFICATION_THRESHOLD,
    POSITIVE_CLASS,
)


def evaluate_classifier(
    pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
    experiment_name,
    model_name,
    threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
):

    pipeline.fit(X_train, y_train)

    train_preds = pipeline.predict(X_train)

    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)

    training_accuracy = accuracy_score(y_train, train_preds)
    testing_accuracy = accuracy_score(y_test, test_preds)

    support_precision = precision_score(y_test, test_preds, pos_label=POSITIVE_CLASS)
    support_recall = recall_score(y_test, test_preds, pos_label=POSITIVE_CLASS)
    support_f1 = f1_score(y_test, test_preds, pos_label=POSITIVE_CLASS)
    roc_auc = roc_auc_score(y_test, test_probs)

    cm = confusion_matrix(y_test, test_preds, labels=[0, 1])

    result = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "training_accuracy": training_accuracy,
        "testing_accuracy": testing_accuracy,
        "support_precision": support_precision,
        "support_recall": support_recall,
        "support_f1": support_f1,
        "roc_auc": roc_auc,
        "cv_mean": np.nan,
        "cv_std": np.nan,
        "confusion_matrix": cm,
    }

    return result


def cross_validate_classifier(pipeline, X_train, y_train, primary_metric="recall"):

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    scoring = {"recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}

    cv_results = cross_validate(
        pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
    )

    summary = {
        "recall_mean": cv_results["test_recall"].mean(),
        "recall_std": cv_results["test_recall"].std(),
        "f1_mean": cv_results["test_f1"].mean(),
        "f1_std": cv_results["test_f1"].std(),
        "roc_auc_mean": cv_results["test_roc_auc"].mean(),
        "roc_auc_std": cv_results["test_roc_auc"].std(),
    }

    summary["cv_mean"] = summary[f"{primary_metric}_mean"]
    summary["cv_std"] = summary[f"{primary_metric}_std"]

    return summary


def attach_cv_results(result, cv_summary):
    result["cv_mean"] = cv_summary["cv_mean"]
    result["cv_std"] = cv_summary["cv_std"]
    return result


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    labels = ["Likely to pass", "May need support"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(title)
    plt.show()


def compare_model_results(results_list):
    df = pd.DataFrame(results_list)
    df = df.sort_values(by=["support_recall", "support_f1", "roc_auc"], ascending=False)
    df = df.reset_index(drop=True)
    return df


def evaluate_subgroups(
    pipeline,
    X_test,
    y_test,
    subgroup_column,
    group_name,
    threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
    min_group_size=10,
):

    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    df = pd.DataFrame(
        {
            "group": subgroup_column.values,
            "y_true": y_test.values,
            "y_pred": preds,
        }
    )

    rows = []
    for group_value, group_df in df.groupby("group"):
        rows.append(
            {
                "subgroup_variable": group_name,
                "group": group_value,
                "count": len(group_df),
                "accuracy": accuracy_score(group_df["y_true"], group_df["y_pred"]),
                "support_precision": precision_score(
                    group_df["y_true"],
                    group_df["y_pred"],
                    pos_label=POSITIVE_CLASS,
                    zero_division=0,
                ),
                "support_recall": recall_score(
                    group_df["y_true"],
                    group_df["y_pred"],
                    pos_label=POSITIVE_CLASS,
                    zero_division=0,
                ),
                "support_f1": f1_score(
                    group_df["y_true"],
                    group_df["y_pred"],
                    pos_label=POSITIVE_CLASS,
                    zero_division=0,
                ),
                "low_sample_warning": len(group_df) < min_group_size,
            }
        )

    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
