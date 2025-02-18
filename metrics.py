import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, TSNE

# -----------------------------------------------------------------------------
# 1. One-vs-Rest Classification Consistency
# -----------------------------------------------------------------------------
def classification_consistency_score(X_low, y, train_size=0.7, random_state=None):
    """
    If y is continuous, discretize it into 10 equal-width intervals.
    Then, perform a train-test split on X_low and the discretized y,
    train a linear SVC (one-vs-rest), and return the test accuracy.
    """
    from sklearn.preprocessing import StandardScaler
    
    # Normalize the embedding
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_low)
    
    # Discretize continuous labels
    if np.issubdtype(y.dtype, np.floating):
        bins = np.linspace(np.min(y), np.max(y), 11)
        y_disc = np.digitize(y, bins) - 1  # yields classes 0 to 9
    else:
        y_disc = y

    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_disc, train_size=train_size, random_state=random_state
    )
    clf = SVC(kernel='linear', decision_function_shape='ovr', random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# -----------------------------------------------------------------------------
# 2. Trustworthiness
# -----------------------------------------------------------------------------
def trustworthiness_score(X_high, X_low, n_neighbors=7):
    """
    Computes the trustworthiness metric:
      T = 1 - (2/(n * k * (2n - 3k - 1))) * sum_{i=1}^n sum_{j in U(i)} (r(i, j) - k)
    where U(i) is the set of points that are among the k nearest neighbors in the
    embedding but not in the original space, and r(i,j) is the rank of j in the
    original space for point i.
    """
    n_samples = X_high.shape[0]
    if n_neighbors >= n_samples:
        raise ValueError("n_neighbors must be less than the number of samples.")
    
    nbrs_high = NearestNeighbors(n_neighbors=n_samples).fit(X_high)
    distances_high, indices_high = nbrs_high.kneighbors(X_high)

    nbrs_low = NearestNeighbors(n_neighbors=n_neighbors).fit(X_low)
    _, indices_low = nbrs_low.kneighbors(X_low)

    total = 0.0
    for i in range(n_samples):
        # True neighbors from the original space (exclude self: first neighbor is i itself)
        true_neighbors = set(indices_high[i, 1:n_neighbors+1])
        embedded_neighbors = set(indices_low[i])
        U = embedded_neighbors - true_neighbors
        for j in U:
            # Get the rank of j in the original ordering (0-indexed, so add 1 for 1-indexed rank)
            rank = np.where(indices_high[i] == j)[0][0] + 1
            total += (rank - n_neighbors)
    factor = 2.0 / (n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1))
    T = 1 - factor * total
    return T

# -----------------------------------------------------------------------------
# 3. Continuity
# -----------------------------------------------------------------------------
def continuity_score(X_high, X_low, n_neighbors=7):
    """
    Computes the continuity metric:
      C = 1 - (2/(n * k * (2n - 3k - 1))) * sum_{i=1}^n sum_{j in V(i)} (r'(i, j) - k)
    where V(i) is the set of points that are among the k nearest neighbors in the
    original space but missing in the embedded space, and r'(i,j) is the rank of j in the
    embedded space for point i.
    """
    n_samples = X_high.shape[0]
    if n_neighbors >= n_samples:
        raise ValueError("n_neighbors must be less than the number of samples.")

    nbrs_low_all = NearestNeighbors(n_neighbors=n_samples).fit(X_low)
    distances_low_all, indices_low_all = nbrs_low_all.kneighbors(X_low)

    nbrs_high = NearestNeighbors(n_neighbors=n_neighbors).fit(X_high)
    _, indices_high = nbrs_high.kneighbors(X_high)

    total = 0.0
    for i in range(n_samples):
        true_neighbors = set(indices_high[i])
        # In the embedded space, exclude self (first index)
        preserved = set(indices_low_all[i, 1:n_neighbors+1])
        V = true_neighbors - preserved
        for j in V:
            rank = np.where(indices_low_all[i] == j)[0][0]  # 0-indexed; assume self is at index 0
            total += (rank - n_neighbors)
    factor = 2.0 / (n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1))
    C = 1 - factor * total
    return C

import pandas as pd


def evaluate_methods_on_datasets(best_isomap, best_LLE, datasets, precomputed=False):
    """
    ... (docstring reste identique) ...
    """
    import time
    results = {}
    methods = ['mds', 'isomap', 'lle', 't-sne']
    X_low_precomputed = {}

    for dataset_name, (X, y) in datasets.items():
        results[dataset_name] = {}
        X_low_precomputed[dataset_name] = {}
        
        # Discretize continuous labels if needed
        if np.issubdtype(y.dtype, np.number) and len(np.unique(y)) > 20:
            y_discrete = pd.qcut(y, q=10, labels=False)
        else:
            y_discrete = y
        
        for method in methods:
            # Compute the low-dimensional representation (X_low) for each method:
            start_time = time.time()  # Start timing
            
            if method == 'mds':
                X_low = MDS(n_components=2, random_state=42).fit_transform(X)
            elif method == 'isomap':
                n_neighbors_iso = best_isomap.get(dataset_name, 7)
                X_low = Isomap(n_neighbors=n_neighbors_iso, n_components=2).fit_transform(X)
            elif method == 'lle':
                n_neighbors_lle = best_LLE.get(dataset_name, 7)
                X_low = LocallyLinearEmbedding(n_neighbors=n_neighbors_lle, n_components=2).fit_transform(X)
            elif method == 't-sne':
                X_low = TSNE(n_components=2, random_state=42).fit_transform(X)
            else:
                continue

            X_low_precomputed[dataset_name][method] = (X_low, y_discrete)
            computation_time = time.time() - start_time  # End timing

            # Compute metrics
            metrics_dict = {}
            metrics_dict['computation_time'] = computation_time  # Add computation time
            metrics_dict['trustworthiness'] = trustworthiness_score(X, X_low, n_neighbors=7)
            metrics_dict['continuity'] = continuity_score(X, X_low, n_neighbors=7)
            metrics_dict['classification_consistency'] = classification_consistency_score(
                X_low, y_discrete, train_size=0.7, random_state=42
            )
            results[dataset_name][method] = metrics_dict
    if precomputed:
        return results, X_low_precomputed
    else:
        return results


def aggregate_metrics(results):
    """
    Given the results dict with quality metrics for each method on each dataset,
    compute the overall aggregated quality score Θ for every method according to:

         Θ = ( (1 - M_t) + (1 - M_c) + (1 - M_class) ) / 3,

    Lower Θ (closer to 0) is better.

    Parameters:
        results: dict with structure:
            {
                dataset_name: {
                    method_name: {
                        metric_name: value
                    }
                }
            }
    
    Returns:
        aggregated_results: dict with structure:
            {
                dataset_name: {
                    method_name: aggregated_score
                }
            }
    """
    aggregated_results = {}
    
    for dataset_name, methods_dict in results.items():
        aggregated_results[dataset_name] = {}
        
        for method_name, metrics_dict in methods_dict.items():
            M_t = metrics_dict['trustworthiness']
            M_c = metrics_dict['continuity']
            M_class = metrics_dict['classification_consistency']
            Theta = ((1 - M_t) + (1 - M_c) + (1 - M_class) ) / 3
            aggregated_results[dataset_name][method_name] = Theta

    return aggregated_results
