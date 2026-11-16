
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
    davies_bouldin_score
)


def evaluate_classification(y_true, y_pred):
    """Evaluate classification model performance."""

    cm = confusion_matrix(y_true, y_pred)

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0
        )
    }

    # Calculate FP and FN for binary classification
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()

        results["true_negative"] = int(tn)
        results["false_positive"] = int(fp)
        results["false_negative"] = int(fn)
        results["true_positive"] = int(tp)

    return results


def evaluate_regression(y_true, y_pred):
    """Evaluate regression model performance."""

    mse = mean_squared_error(y_true, y_pred)

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2_score": r2_score(y_true, y_pred)
    }


def evaluate_clustering(X, labels):
    """Evaluate clustering model performance."""

    unique_labels = set(labels)

    # Silhouette score requires at least 2 clusters
    # and fewer clusters than observations.
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return {
            "silhouette_score": None,
            "davies_bouldin_score": None,
            "message": "Not enough valid clusters for evaluation."
        }

    return {
        "silhouette_score": silhouette_score(X, labels),
        "davies_bouldin_score": davies_bouldin_score(X, labels)
    }
