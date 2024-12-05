import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_olivetti_faces
from PIL import Image
def load_fashion_mnist(n_points=1000):
    """Charge les données Fashion-MNIST."""
    fashion_mnist = fetch_openml('Fashion-MNIST', version=1)
    data = fashion_mnist.data
    labels = fashion_mnist.target.astype(int)
    # Réduire à 1000 points en gardant la répartition des classes
    data, _, labels, _ = train_test_split(data, labels, train_size=n_points, stratify=labels, random_state=42)
    return data, labels

def generate_linear_data(n_points=1000, n_dim=4, hole_size=0.5):
    """
    Génère un jeu de données en forme de Swiss Roll en n dimensions avec des trous,
    avec enroulement dans toutes les dimensions et en augmentant la sharpness de l'enroulement.
    
    Parameters:
    -----------
    n_points : int
        Nombre de points à générer
    n_dim : int
        Nombre de dimensions
    hole_size : float
        Taille relative des trous (entre 0 et 1)
    """
    # Générer plus de points initialement pour compenser ceux qui seront retirés
    n_points_initial = int(n_points * 1.5)
    t = np.linspace(0, 3*np.pi, n_points_initial)
    
    # Créer le masque pour les trous
    mask = np.ones_like(t, dtype=bool)
    for hole_center in [np.pi, 2*np.pi]:
        hole_mask = np.abs(t - hole_center) < (hole_size * np.pi/4)
        mask = mask & ~hole_mask
    
    # Appliquer le masque à t
    t = t[mask]
    
    # S'assurer qu'on a le bon nombre de points
    if len(t) > n_points:
        indices = np.random.choice(len(t), n_points, replace=False)
        t = t[indices]
    
    # Générer les coordonnées avec les trous
    coords = [t * np.cos((i + 1) * t + (i * np.pi / n_dim)) if i % 2 == 0 
             else t * np.sin((i + 1) * t + (i * np.pi / n_dim)) 
             for i in range(n_dim)]
    
    data = np.vstack(coords).T
    return data, t


def generate_hyper_swiss_roll(n_points=1000, n_dim=10):
    """Génère un jeu de données en forme de Swiss Roll en n dimensions, avec enroulement dans toutes les dimensions et en augmentant la sharpness de l'enroulement."""
    t = np.linspace(0, 3*np.pi, n_points)
    coords = [t * np.cos((i + 1) * t + (i * np.pi / n_dim)) if i % 2 == 0 else t * np.sin((i + 1) * t + (i * np.pi / n_dim)) for i in range(n_dim)]
    data = np.vstack(coords).T
    return data, t

def generate_spiral_with_bridges(n_points=1000, noise=0.1, n_bridges=3):
    """
    Génère deux spirales connectées par des ponts.
    """
    # Calculer le nombre de points par spirale en tenant compte des ponts
    points_for_bridges = 20 * n_bridges
    points_per_spiral = (n_points - points_for_bridges) // 2
    
    # Générer la première spirale
    t = np.linspace(0, 4*np.pi, points_per_spiral)
    r = t
    x1 = r * np.cos(t)
    y1 = r * np.sin(t)
    z1 = t
    spiral1 = np.column_stack([x1, y1, z1])
    
    # Générer la deuxième spirale (décalée)
    x2 = r * np.cos(t) + 10
    y2 = r * np.sin(t) + 10
    z2 = t
    spiral2 = np.column_stack([x2, y2, z2])
    
    # Combiner les spirales
    data = np.vstack([spiral1, spiral2])
    
    # Ajouter des ponts entre les spirales
    bridge_points_list = []
    for i in range(n_bridges):
        idx1 = (i + 1) * len(spiral1) // (n_bridges + 1)
        idx2 = (i + 1) * len(spiral2) // (n_bridges + 1)
        
        # Créer des points intermédiaires pour le pont
        bridge_points = np.linspace(data[idx1], data[idx2], num=20)
        bridge_points_list.append(bridge_points)
    
    # Ajouter tous les points des ponts
    data = np.vstack([data] + bridge_points_list)
    
    # Ajouter du bruit
    data += np.random.normal(0, noise, data.shape)
    
    # Créer des labels pour la visualisation
    labels = np.arange(len(data)) / len(data)
    
    assert len(data) == n_points, f"Expected {n_points} points, got {len(data)}"
    assert len(labels) == n_points, f"Expected {n_points} labels, got {len(labels)}"
    
    return data, labels

