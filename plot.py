import matplotlib.pyplot as plt
import numpy as np
from reduction_methods import apply_dimensionality_reduction
from sklearn.manifold import TSNE, Isomap, MDS, LocallyLinearEmbedding

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

def plot_3d_and_2d_projection(dataset, labels, title="Dataset"):
    """
    Visualise un jeu de données 3D et sa projection 2D.
    
    Parameters:
    -----------
    dataset : numpy.ndarray
        Données 3D de forme (n_points, 3)
    labels : numpy.ndarray 
        Étiquettes pour colorer les points
    title : str
        Titre de base pour les graphiques
    """
    fig = plt.figure(figsize=(12, 6))
    
    # Visualisation 3D
    ax3d = fig.add_subplot(121, projection='3d')
    scatter = ax3d.scatter(dataset[:,0], dataset[:,1], dataset[:,2], 
                          c=labels, cmap='viridis')
    fig.colorbar(scatter, ax=ax3d, label='Progression')

    ax3d.set_title(f"Trajectoire {title} 3D")
    ax3d.set_xlabel("Dimension 0")
    ax3d.set_ylabel("Dimension 1")
    ax3d.set_zlabel("Dimension 2")

    # Projection 2D sur le plan (0,1)
    ax2d = fig.add_subplot(122)
    proj_2d = dataset[:, [0,1]]
    scatter_2d = ax2d.scatter(proj_2d[:, 0], proj_2d[:, 1], 
                             c=labels, cmap='viridis')
    fig.colorbar(scatter_2d, ax=ax2d, label='Progression')

    ax2d.set_title(f"Projection 2D de {title} sur (0,1)")
    ax2d.set_xlabel("Dimension 0")
    ax2d.set_ylabel("Dimension 1")
    ax2d.grid(True)

    plt.tight_layout()
    plt.show()

def plot_dimensionality_reduction_results(dataset, labels=None, n_neighbors_dict={"LLE": 10, "Isomap": 10}):
    """
    Affiche les résultats de la réduction de dimensionnalité pour un jeu de données.
    
    Parameters:
    -----------
    dataset : numpy.ndarray
        Le jeu de données d'origine à réduire
    labels : numpy.ndarray, optional
        Les étiquettes pour colorer les points. Si None, utilise l'ordre des points.
    """
    # Appliquer la réduction de dimension
    tsne_result, lle_result, isomap_result, mds_result = apply_dimensionality_reduction(dataset, n_neighbors_dict)
    
    if labels is None:
        labels = np.arange(len(dataset))
    
    # Créer des sous-graphiques pour afficher les résultats
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Dictionnaire des résultats et leurs positions/titres
    results = {
        (0, 0): ('t-SNE', tsne_result),
        (0, 1): ('LLE', lle_result),
        (1, 0): ('Isomap', isomap_result),
        (1, 1): ('MDS', mds_result)
    }
    
    # Afficher chaque méthode
    for (i, j), (method, result) in results.items():
        axes[i, j].scatter(result[:, 0], result[:, 1], 
                          c=labels, cmap='viridis', s=10)
        axes[i, j].set_title(method)
    
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def plot_embeddings(embeddings, dataset_names=None, methods=None, figsize=(12, 12)):
    """
    Plots a 2x2 grid of embeddings for each dataset with all methods overlaid.
    
    Parameters:
      embeddings: dict
          Dictionary with keys as dataset names and values as dicts mapping method names
          to 2D numpy arrays of shape (n_samples, 2). For example:
          {
              'Helicoids': {'mds': X_low_mds, 'isomap': X_low_isomap, 'lle': X_low_lle, 't-sne': X_low_tsne},
              'Zigzag': { ... },
              ...
          }
      dataset_names: list (optional)
          List of dataset names to plot. If None, all keys in embeddings are used.
      methods: list (optional)
          List of method names to plot. If None, defaults to ['mds', 'isomap', 'lle', 't-sne'].
      figsize: tuple (optional)
          Figure size for the plot.
    """
    # Default methods if not provided.
    if methods is None:
        methods = ['mds', 'isomap', 'lle', 't-sne']
    
    # Use all dataset keys if dataset_names not provided.
    if dataset_names is None:
        dataset_names = list(embeddings.keys())
    
    # Define colors for each method.
    colors = {
        'mds': 'red',
        'isomap': 'blue',
        'lle': 'green',
        't-sne': 'purple'
    }
    
    # Create a 2x2 grid of subplots.
    n_datasets = len(dataset_names)
    n_cols = 2
    n_rows = int(np.ceil(n_datasets / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = axs.flatten()  # Flatten in case of more than one row.
    
    # Loop over datasets and plot embeddings for each method.
    for i, ds in enumerate(dataset_names):
        ax = axs[i]
        for m in methods:
            if m in embeddings[ds]:
                X_low = embeddings[ds][m]
                ax.scatter(X_low[:, 0], X_low[:, 1], s=20, color=colors.get(m, 'black'),
                           alpha=0.6, label=m)
        ax.set_title(ds)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.legend(loc='best')
    
    # Hide any unused subplots.
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
    
    plt.tight_layout()
    plt.show()

# # Example usage:
# if __name__ == '__main__':
#     # Dummy embeddings for demonstration.
#     np.random.seed(42)
#     embeddings = {}
#     dataset_names = ['Helicoids', 'Zigzag', 'Swiss Roll', 'Bonhomme']
#     methods = ['mds', 'isomap', 'lle', 't-sne']
    
#     for ds in dataset_names:
#         embeddings[ds] = {}
#         for m in methods:
#             # Replace with actual embeddings; here we generate random points.
#             embeddings[ds][m] = np.random.rand(100, 2)
    
#     plot_embeddings(embeddings)


import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, TSNE

def plot_all_methods_and_datasets(datasets, best_isomap, best_lle, precomputed=None):
    """
    Plots a grid of embeddings where each row corresponds to a DR method 
    (t-SNE, LLE, Isomap, MDS) and each column to a dataset.
    
    - datasets: dict of {dataset_name: (X, y)}
    - best_isomap, best_lle: dicts mapping dataset_name -> best K
    - precomputed (optional): dict of {dataset_name: {method: (X_reduced, y), ...}}
      If provided, the embedding for a given dataset and method is used directly.
    - Ticks are removed and large, bold labels are used for clarity.
    
    When only one dataset is provided, the four methods are arranged in a 2x2 grid.
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE, Isomap, MDS
    from sklearn.manifold import LocallyLinearEmbedding

    methods_order = ["t-sne", "lle", "isomap", "mds"]
    titles = {
        "t-sne":   "t-SNE",
        "lle":     "LLE",
        "isomap":  "Isomap",
        "mds":     "MDS"
    }
    
    dataset_names = list(datasets.keys())
    
    # If there is only one dataset, arrange plots in a 2x2 grid.
    if len(dataset_names) == 1:
        n_rows, n_cols = 2, 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
        axes = axes.flatten()
        
        dataset_name = dataset_names[0]
        X, y = datasets[dataset_name]
        
        for idx, method in enumerate(methods_order):
            # Check if precomputed embeddings are available.
            if precomputed is not None and dataset_name in precomputed and method in precomputed[dataset_name]:
                X_embedded, y = precomputed[dataset_name][method]
            else:
                # Compute embedding if not precomputed.
                if method == 't-sne':
                    embedder = TSNE(n_components=2, random_state=42)
                elif method == 'lle':
                    k = best_lle[dataset_name]
                    embedder = LocallyLinearEmbedding(n_neighbors=k, n_components=2, random_state=42)
                elif method == 'isomap':
                    k = best_isomap[dataset_name]
                    embedder = Isomap(n_neighbors=k, n_components=2)
                elif method == 'mds':
                    embedder = MDS(n_components=2, random_state=42)
                else:
                    raise ValueError(f"Unknown method {method}")
                X_embedded = embedder.fit_transform(X)
            
            ax = axes[idx]
            ax.scatter(X_embedded[:, 0], X_embedded[:, 1],
                       c=y, cmap='viridis')
            ax.set_title(f"{titles[method]}\n({dataset_name})", fontweight='bold', fontsize=18)
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide any unused subplots if present.
        for j in range(len(methods_order), len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.show()
        
    else:
        n_rows = len(methods_order)
        n_cols = len(dataset_names)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        for row, method in enumerate(methods_order):
            for col, dataset_name in enumerate(dataset_names):
                X, y = datasets[dataset_name]
                
                # Select the appropriate axis.
                if n_rows == 1 and n_cols == 1:
                    ax = axes
                elif n_rows == 1:
                    ax = axes[col]
                elif n_cols == 1:
                    ax = axes[row]
                else:
                    ax = axes[row, col]
                    
                # Use precomputed embedding if available.
                if precomputed is not None and dataset_name in precomputed and method in precomputed[dataset_name]:
                    X_embedded, y = precomputed[dataset_name][method]
                else:
                    if method == 't-sne':
                        embedder = TSNE(n_components=2, random_state=42)
                    elif method == 'lle':
                        k = best_lle[dataset_name]
                        embedder = LocallyLinearEmbedding(n_neighbors=k, n_components=2, random_state=42)
                    elif method == 'isomap':
                        k = best_isomap[dataset_name]
                        embedder = Isomap(n_neighbors=k, n_components=2)
                    elif method == 'mds':
                        embedder = MDS(n_components=2, random_state=42)
                    else:
                        raise ValueError(f"Unknown method {method}")
                    X_embedded = embedder.fit_transform(X)
                
                ax.scatter(X_embedded[:, 0], X_embedded[:, 1],
                           c=y, cmap='viridis')
                
                if row == 0:
                    ax.set_title(dataset_name, fontweight='bold', fontsize=30)
                if col == 0:
                    ax.set_ylabel(titles[method], fontweight='bold', fontsize=40)
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.suptitle("Grid of 2D embeddings (rows=methods, columns=datasets)",
                     fontweight='bold', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
