
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def calculate_metrics(y_true, y_pred):
    """
    Calculate accuracy, macro precision, macro recall,
    and macro F1-score.
    """

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro Recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro F1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )
    }


def evaluate_model(model, dataset):
    """
    Evaluate a Keras model on a dataset yielding
    (images, integer_labels).

    Returns metrics and true/predicted labels.
    """

    y_true = []
    y_pred = []

    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)

        y_true.extend(labels.numpy().astype(int))
        y_pred.extend(np.argmax(predictions, axis=1))

    metrics = calculate_metrics(y_true, y_pred)

    return {
        "metrics": metrics,
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred)
    }


def get_classification_report(y_true, y_pred):
    """
    Return a per-class classification report as a DataFrame.
    """

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    return pd.DataFrame(report).transpose()


def get_confusion_matrix(y_true, y_pred):
    """
    Return a confusion matrix.
    """

    return confusion_matrix(y_true, y_pred)
