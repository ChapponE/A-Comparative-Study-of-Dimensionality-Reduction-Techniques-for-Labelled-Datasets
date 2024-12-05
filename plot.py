import matplotlib.pyplot as plt
from reduction_methods import apply_tsne, apply_lle, apply_isomap, apply_mds

def plot_reducted_data(data, labels, method_names, n_components=2, is_helix=False):
    """Compare différentes méthodes de réduction de dimension et affiche les résultats."""
    
    # Check if n_components is valid
    if n_components not in [2, 3]:
        raise ValueError("n_components must be either 2 or 3.")
    
    methods = {
        "t-SNE": apply_tsne,
        "LLE": apply_lle,
        "Isomap": apply_isomap,
        "MDS": apply_mds,
    }

    plt.figure(figsize=(15, 10))
    for i, method_name in enumerate(method_names):
        reduced_data = methods[method_name](data, n_components=n_components)
        ax = plt.subplot(2, 2, i + 1, projection='3d' if n_components == 3 else None)
        if n_components == 2:
            if is_helix:
                ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.7)
            else:
                ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='tab10', alpha=0.7)
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
        else:
            if is_helix:
                ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=labels, cmap='viridis', alpha=0.7)
            else:
                ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=labels, cmap='tab10', alpha=0.7)
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
        ax.set_title(method_name)

    plt.tight_layout()
    plt.show()



