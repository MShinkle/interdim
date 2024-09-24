from typing import Literal, Optional, Union

import numpy as np
from sklearn import metrics

ScoreMethodType = Literal[
    "adjusted_mutual_info",
    "adjusted_rand",
    "completeness",
    "fowlkes_mallows",
    "homogeneity",
    "mutual_info",
    "normalized_mutual_info",
    "rand",
    "v_measure",
    "contingency_matrix",
    "pair_confusion_matrix",
    "calinski_harabasz",
    "davies_bouldin",
    "silhouette",
]


def score_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    method: ScoreMethodType = "silhouette",
) -> Union[float, np.ndarray]:
    """
    Evaluate clustering performance using the specified method.

    Args:
        X: The input data.
        labels: The predicted cluster labels.
        true_labels: The true cluster labels (optional).
        method: The evaluation method to use.

    Returns:
        The computed metric (float or array).

    Raises:
        ValueError: If an unsupported evaluation method is specified or if
                    true labels are not provided for methods that require them.
    """
    method = method.lower()

    if method in [
        "adjusted_mutual_info",
        "adjusted_rand",
        "completeness",
        "fowlkes_mallows",
        "homogeneity",
        "mutual_info",
        "normalized_mutual_info",
        "rand",
        "v_measure",
        "contingency_matrix",
        "pair_confusion_matrix",
    ]:
        if true_labels is None:
            raise ValueError(f"The {method} method requires true labels.")
        if len(true_labels) != len(labels):
            raise ValueError("Number of true labels does not match number of samples")

    if method == "adjusted_mutual_info":
        return metrics.adjusted_mutual_info_score(true_labels, labels)
    elif method == "adjusted_rand":
        return metrics.adjusted_rand_score(true_labels, labels)
    elif method == "calinski_harabasz":
        return metrics.calinski_harabasz_score(X, labels)
    elif method == "contingency_matrix":
        return metrics.cluster.contingency_matrix(true_labels, labels)
    elif method == "pair_confusion_matrix":
        return metrics.cluster.pair_confusion_matrix(true_labels, labels)
    elif method == "completeness":
        return metrics.completeness_score(true_labels, labels)
    elif method == "davies_bouldin":
        return metrics.davies_bouldin_score(X, labels)
    elif method == "fowlkes_mallows":
        return metrics.fowlkes_mallows_score(true_labels, labels)
    elif method == "homogeneity":
        return metrics.homogeneity_score(true_labels, labels)
    elif method == "mutual_info":
        return metrics.mutual_info_score(true_labels, labels)
    elif method == "normalized_mutual_info":
        return metrics.normalized_mutual_info_score(true_labels, labels)
    elif method == "rand":
        return metrics.rand_score(true_labels, labels)
    elif method == "silhouette":
        return metrics.silhouette_score(X, labels)
    elif method == "v_measure":
        return metrics.v_measure_score(true_labels, labels)
    else:
        raise ValueError(f"Unsupported evaluation method: {method}")
