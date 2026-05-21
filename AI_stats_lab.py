"""
AI_stats_lab.py

Lab: Unsupervised Learning and K-Means Clustering

Topics:
- Unsupervised learning with unlabeled data
- Iris dataset without labels
- Feature standardization
- K-Means clustering
- K-Means objective function
- Elbow method for choosing K
- Underfitting and overfitting in clustering
- Distance-based outlier detection
- Visualization of unlabeled data, clusters, centroids, and elbow curve

Instructions:
- Implement all functions.
- Do NOT change function names.
- Do NOT print inside functions.
- Return exactly the required formats.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


# ============================================================
# Question 1: Unlabeled Data and K-Means Clustering
# ============================================================

def load_iris_unlabeled(feature_indices=(0, 1)):
    """
    Load the Iris dataset without labels.
    """
    iris = load_iris()
    X = iris.data[:, list(feature_indices)]
    feature_names = [iris.feature_names[i] for i in feature_indices]
    return {
        "X": X,
        "feature_names": feature_names
    }


def standardize_features(X):
    """
    Standardize features to zero mean and unit variance.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Replace zero std with 1 to avoid division by zero
    std[std == 0] = 1.0
    X_scaled = (X - mean) / std
    return {
        "X_scaled": X_scaled,
        "mean": mean,
        "std": std
    }


def fit_kmeans(X, K, random_state=0, n_init=10):
    """
    Fit K-Means clustering on data X.
    """
    model = KMeans(n_clusters=K, random_state=random_state, n_init=n_init)
    model.fit(X)
    return {
        "centroids": model.cluster_centers_,
        "labels": model.labels_,
        "objective": model.inertia_,
        "n_iter": model.n_iter_
    }


def compute_kmeans_objective(X, centroids, labels):
    """
    Compute the K-Means objective manually.
    """
    objective = 0.0
    for i, x in enumerate(X):
        c = centroids[labels[i]]
        diff = x - c
        objective += np.dot(diff, diff)
    return objective


# ============================================================
# Question 2: Choosing K, Underfitting/Overfitting, and Outliers
# ============================================================

def evaluate_k_values(X, k_values, random_state=0, n_init=10):
    """
    Run K-Means for multiple values of K.
    """
    objectives = []
    for K in k_values:
        result = fit_kmeans(X, K, random_state=random_state, n_init=n_init)
        objectives.append(result["objective"])

    relative_improvements = [0.0]
    for i in range(1, len(objectives)):
        prev = objectives[i - 1]
        curr = objectives[i]
        if prev == 0:
            relative_improvements.append(0.0)
        else:
            relative_improvements.append((prev - curr) / prev)

    return {
        "k_values": k_values,
        "objectives": objectives,
        "relative_improvements": relative_improvements
    }


def choose_elbow_k(k_values, objectives):
    """
    Choose K using the maximum-distance-to-line heuristic (elbow method).
    """
    if len(k_values) < 3:
        return k_values[0]

    # First and last points
    x1, y1 = k_values[0], objectives[0]
    x2, y2 = k_values[-1], objectives[-1]

    # Direction vector of the line
    dx = x2 - x1
    dy = y2 - y1
    line_len = np.sqrt(dx**2 + dy**2)

    max_dist = -1
    best_k = k_values[0]

    for i in range(1, len(k_values) - 1):
        x0, y0 = k_values[i], objectives[i]
        # Perpendicular distance from point to line
        dist = abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / line_len
        if dist > max_dist:
            max_dist = dist
            best_k = k_values[i]

    return best_k


def cluster_size_summary(labels, K):
    """
    Count how many data points belong to each cluster.
    """
    summary = {}
    for k in range(K):
        summary[k] = int(np.sum(labels == k))
    return summary


def identify_outliers_by_distance(X, centroids, labels, top_n=5):
    """
    Identify possible outliers based on distance from assigned centroid.
    """
    distances = np.array([
        np.dot(X[i] - centroids[labels[i]], X[i] - centroids[labels[i]])
        for i in range(len(X))
    ])
    sorted_indices = np.argsort(distances)[::-1]
    top_indices = sorted_indices[:top_n]
    top_distances = distances[top_indices]
    return {
        "indices": top_indices,
        "distances": top_distances
    }


def diagnose_clustering_fit(K, elbow_k):
    """
    Diagnose whether the chosen K is underfitting, good fit, or overfitting.
    """
    if K < elbow_k:
        return "underfitting"
    elif K == elbow_k:
        return "good_fit"
    else:
        return "overfitting"


# ============================================================
# Question 3: Visualization
# ============================================================

def plot_unlabeled_data(X, feature_names=None, title="Unlabeled Data"):
    """
    Visualize unlabeled 2D data.
    """
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], alpha=0.6)
    ax.set_title(title)
    if feature_names is not None:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
    return fig, ax


def plot_kmeans_clusters(X, labels, centroids, feature_names=None, title="K-Means Clusters"):
    """
    Visualize K-Means clustering results.
    """
    fig, ax = plt.subplots()
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        ax.scatter(X[mask, 0], X[mask, 1], label=f"Cluster {label}", alpha=0.6)
    ax.scatter(
        centroids[:, 0], centroids[:, 1],
        marker="X", s=200, c="black", zorder=5, label="Centroids"
    )
    ax.set_title(title)
    if feature_names is not None:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
    ax.legend()
    return fig, ax


def plot_elbow_curve(k_values, objectives, title="Elbow Method"):
    """
    Plot K-Means objective values versus K.
    """
    fig, ax = plt.subplots()
    ax.plot(k_values, objectives, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Number of clusters K")
    ax.set_ylabel("Objective value")
    return fig, ax


if __name__ == "__main__":
    print("Implement all required functions.")
