import matplotlib.pyplot as plt
from reduction_methods import apply_tsne, apply_lle, apply_isomap, apply_mds

def plot_reducted_data(data, labels, method_names, n_components=2, is_helix=False, n_neighbors=None):
    """Compare différentes méthodes de réduction de dimension et affiche les résultats.
    
    Parameters:
    -----------
    data : array-like, shape (n_samples, n_features)
        The input high-dimensional data.
    labels : array-like, shape (n_samples,)
        The labels for coloring the data points.
    method_names : list of str
        List of method names to apply (e.g., ["t-SNE", "LLE", "Isomap", "MDS"]).
    n_components : int, optional (default=2)
        Number of dimensions for the reduced data.
    is_helix : bool, optional (default=False)
        Whether the dataset is a helix, affecting the color map.
    n_neighbors : int or dict, optional (default=None)
        Number of neighbors to use for methods that require it (e.g., LLE, Isomap).
        Can be a single integer or a dictionary mapping method names to integers.
    """
    
    methods = {
        "t-SNE": apply_tsne,
        "LLE": apply_lle,
        "Isomap": apply_isomap,
        "MDS": apply_mds,
    }

    plt.figure(figsize=(15, 10))
    for i, method_name in enumerate(method_names):
        if method_name not in methods:
            raise ValueError(f"Method '{method_name}' is not supported.")
        
        # Determine the appropriate n_neighbors for the method
        current_n_neighbors = None
        if method_name in ['LLE', 'Isomap']:
            if isinstance(n_neighbors, dict):
                current_n_neighbors = n_neighbors.get(method_name, 10)  # Default to 10 if not specified
            elif isinstance(n_neighbors, int):
                current_n_neighbors = n_neighbors
            else:
                current_n_neighbors = 10  # Default value
        
        # Apply the dimensionality reduction method
        if current_n_neighbors is not None:
            reduced_data = methods[method_name](data, n_components=n_components, n_neighbors=current_n_neighbors)
        else:
            reduced_data = methods[method_name](data, n_components=n_components)
        
        # Create the subplot
        ax = plt.subplot(2, 2, i + 1, projection='3d' if n_components == 3 else None)
        
        # Plot the reduced data
        if n_components == 2:
            cmap = 'viridis' if is_helix else 'tab10'
            ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap=cmap, alpha=0.7)
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
        else:
            cmap = 'viridis' if is_helix else 'tab10'
            ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=labels, cmap=cmap, alpha=0.7)
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
        
        ax.set_title(method_name)

    plt.tight_layout()
    plt.show()


def plot_error_vs_n_neighbors(n_neighbors_results, n_neighbors_range):
    datasets = ['Fashion MNIST', 'Hyper Swiss Roll', 'Double Helix', 'Partial Sphere']
    
    fig, axes = plt.subplots(len(datasets), 2, figsize=(16, 24), sharex=False, sharey=False)

    for i, dataset in enumerate(datasets):
        # LLE
        ax_left = axes[i, 0]
        try:
            ldc_values = n_neighbors_results[dataset]['LLE']['Distance Correlation']
            ax_left.plot(n_neighbors_range, ldc_values, marker='o', color='blue', label='Distance Correlation')
            ax_left.set_title(f"{dataset} - LLE", fontsize=16)
            ax_left.set_xlabel('Nombre de Voisins (n_neighbors)', fontsize=14)
            ax_left.set_ylabel('Distance Correlation', fontsize=14)
            ax_left.legend()
            ax_left.grid(True)
        except KeyError:
            ax_left.text(0.5, 0.5, 'Données manquantes', horizontalalignment='center', verticalalignment='center', fontsize=12)
            ax_left.set_title(f"{dataset} - LLE", fontsize=16)
            ax_left.set_xlabel('Nombre de Voisins (n_neighbors)', fontsize=14)
            ax_left.set_ylabel('Distance Correlation', fontsize=14)
            ax_left.grid(True)

        # Isomap
        ax_right = axes[i, 1]
        try:
            geodesic_errors = n_neighbors_results[dataset]['Isomap']['Geodesic Error']
            ax_right.plot(n_neighbors_range, geodesic_errors, marker='s', color='green', label='Geodesic Error')
            ax_right.set_title(f"{dataset} - Isomap", fontsize=16)
            ax_right.set_xlabel('Nombre de Voisins (n_neighbors)', fontsize=14)
            ax_right.set_ylabel('Geodesic Error', fontsize=14)
            ax_right.legend()
            ax_right.grid(True)
        except KeyError:
            ax_right.text(0.5, 0.5, 'Données manquantes', horizontalalignment='center', verticalalignment='center', fontsize=12)
            ax_right.set_title(f"{dataset} - Isomap", fontsize=16)
            ax_right.set_xlabel('Nombre de Voisins (n_neighbors)', fontsize=14)
            ax_right.set_ylabel('Geodesic Error', fontsize=14)
            ax_right.grid(True)

    plt.tight_layout()
    plt.show()

