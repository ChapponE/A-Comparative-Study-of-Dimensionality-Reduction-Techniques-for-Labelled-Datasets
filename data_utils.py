import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_olivetti_faces
from PIL import Image
from sklearn.datasets import load_wine

import numpy as np
from sklearn.preprocessing import StandardScaler

def load_datasets(n_points=300):
    datasets = {
        'Fashion MNIST': load_fashion_mnist_dataset(n_points=n_points),
        'Hyper Swiss Roll': generate_swiss_roll_dataset(n_points=n_points),
        'Double Helix': generate_double_helix_dataset(n_points=n_points),
        'Partial Sphere': generate_partial_sphere_dataset(n_points=n_points)
    }

    # Standardisation
    datasets = {name: data for name, data in datasets.items()}
    return datasets

# Fashion-MNIST (pour t-sne)
def load_fashion_mnist_dataset(n_points=1000):
    """Charge les données Fashion-MNIST."""
    fashion_mnist = fetch_openml('Fashion-MNIST', version=1)
    data = fashion_mnist.data
    labels = fashion_mnist.target.astype(int)
    data, _, labels, _ = train_test_split(data, labels, train_size=n_points, stratify=labels, random_state=42)
    return data.to_numpy(), labels

# Swiss Roll (pour isomap)
def generate_swiss_roll_dataset(n_points=1000, noise=0.1, random_state=42):
    """
    Génère un jeu de données en forme de Swiss Roll.

    Parameters:
    -----------
    n_points : int
        Nombre de points à générer.
    noise : float
        Niveau de bruit à ajouter aux données.
    random_state : int
        Graine pour la reproductibilité.

    Returns:
    --------
    data : numpy.ndarray
        Données générées, de forme (n_points, 3).
    labels : numpy.ndarray
        Étiquettes correspondant à la position sur le rouleau, de forme (n_points,).
    """
    data, labels = make_swiss_roll(n_samples=n_points, noise=noise, random_state=random_state)
    # Normaliser les données
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data, labels

# Double Helix (pour lle)
def generate_double_helix_dataset(n_points=1000, noise=0.1, random_state=42):
    """
    Génère un jeu de données en forme de Double Helix (double hélice).

    Parameters:
    -----------
    n_points : int
        Nombre total de points à générer (divisible par 2).
    noise : float
        Niveau de bruit à ajouter aux données.
    random_state : int
        Graine pour la reproductibilité.

    Returns:
    --------
    data : numpy.ndarray
        Données générées, de forme (n_points, 3).
    labels : numpy.ndarray
        Étiquettes des hélices, de forme (n_points,).
    """
    np.random.seed(random_state)
    n_points_per_helix = n_points // 2
    t = np.linspace(0, 4 * np.pi, n_points_per_helix)

    # Première hélice
    x1 = np.cos(t)
    y1 = np.sin(t)
    z1 = t
    helix1 = np.vstack([x1, y1, z1]).T

    # Deuxième hélice (décalée de pi)
    x2 = np.cos(t + np.pi)
    y2 = np.sin(t + np.pi)
    z2 = t
    helix2 = np.vstack([x2, y2, z2]).T

    # Combiner les hélices
    data = np.vstack([helix1, helix2])

    # Ajouter du bruit
    data += np.random.normal(scale=noise, size=data.shape)

    # Créer les labels entre 0 et 1
    labels_helix1 = np.linspace(0, 0.5, n_points_per_helix)
    labels_helix2 = np.linspace(0.5, 1, n_points_per_helix)
    labels = np.concatenate([labels_helix1, labels_helix2])

    return data, labels

# Sphere (pour mds)
def generate_partial_sphere_dataset(n_points=1000, noise=0.05, random_state=42):
    """
    Génère un jeu de données en forme de 1/2 de sphère 3D avec des labels continus entre 0 et 1.
    On échantillonne la sphère partiellement, en limitant l'angle polaire phi à [0, 1π/2].
    
    Parameters:
    -----------
    n_points : int
        Nombre total de points à générer.
    noise : float
        Niveau de bruit à ajouter aux données.
    random_state : int
        Graine pour la reproductibilité.
    
    Returns:
    --------
    data : numpy.ndarray
        Données générées, de forme (n_points, 3).
    labels : numpy.ndarray
        Étiquettes des points, de forme (n_points,).
    """
    np.random.seed(random_state)
    
    # Définition des bornes pour phi
    phi_min, phi_max = 0, 1 * np.pi / 2
    cos_min, cos_max = np.cos(phi_max), np.cos(phi_min)
    
    # Échantillonnage uniforme sur la zone sphérique définie
    # On génère cos(phi) uniformément entre [cos(phi_max), cos(phi_min)]
    cos_phi = cos_min + (cos_max - cos_min) * np.random.rand(n_points)
    phi = np.arccos(cos_phi)
    
    theta = 2 * np.pi * np.random.rand(n_points)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    data = np.vstack((x, y, z)).T
    
    # Ajouter du bruit
    data += np.random.normal(scale=noise, size=data.shape)
    
    # Recalculer phi après ajout du bruit pour les labels
    r = np.linalg.norm(data, axis=1)
    phi_noisy = np.arccos(data[:, 2] / r)
    
    # Normaliser les labels entre 0 et 1 en fonction de phi_noisy
    labels = phi_noisy / np.pi
    
    return data, labels

# def generate_linear_data(n_points=1000, n_dim=4, hole_size=0.5):
#     """
#     Génère un jeu de données en forme de Swiss Roll en n dimensions avec des trous,
#     avec enroulement dans toutes les dimensions et en augmentant la sharpness de l'enroulement.
    
#     Parameters:
#     -----------
#     n_points : int
#         Nombre de points à générer
#     n_dim : int
#         Nombre de dimensions
#     hole_size : float
#         Taille relative des trous (entre 0 et 1)
#     """
#     # Générer plus de points initialement pour compenser ceux qui seront retirés
#     n_points_initial = int(n_points * 1.5)
#     t = np.linspace(0, 3*np.pi, n_points_initial)
    
#     # Créer le masque pour les trous
#     mask = np.ones_like(t, dtype=bool)
#     for hole_center in [np.pi, 2*np.pi]:
#         hole_mask = np.abs(t - hole_center) < (hole_size * np.pi/4)
#         mask = mask & ~hole_mask
    
#     # Appliquer le masque à t
#     t = t[mask]
    
#     # S'assurer qu'on a le bon nombre de points
#     if len(t) > n_points:
#         indices = np.random.choice(len(t), n_points, replace=False)
#         t = t[indices]
    
#     # Générer les coordonnées avec les trous
#     coords = [t * np.cos((i + 1) * t + (i * np.pi / n_dim)) if i % 2 == 0 
#              else t * np.sin((i + 1) * t + (i * np.pi / n_dim)) 
#              for i in range(n_dim)]
    
#     data = np.vstack(coords).T
#     return data, t


# def generate_hyper_swiss_roll(n_points=1000, n_dim=10):
#     """Génère un jeu de données en forme de Swiss Roll en n dimensions, avec enroulement dans toutes les dimensions et en augmentant la sharpness de l'enroulement."""
#     t = np.linspace(0, 3*np.pi, n_points)
#     coords = [t * np.cos((i + 1) * t + (i * np.pi / n_dim)) if i % 2 == 0 else t * np.sin((i + 1) * t + (i * np.pi / n_dim)) for i in range(n_dim)]
#     data = np.vstack(coords).T
#     return data, t

# def generate_spiral_with_bridges(n_points=1000, noise=0.1, n_bridges=3):
#     """
#     Génère deux spirales connectées par des ponts.
#     """
#     # Calculer le nombre de points par spirale en tenant compte des ponts
#     points_for_bridges = 20 * n_bridges
#     points_per_spiral = (n_points - points_for_bridges) // 2
    
#     # Générer la première spirale
#     t = np.linspace(0, 4*np.pi, points_per_spiral)
#     r = t
#     x1 = r * np.cos(t)
#     y1 = r * np.sin(t)
#     z1 = t
#     spiral1 = np.column_stack([x1, y1, z1])
    
#     # Générer la deuxième spirale (décalée)
#     x2 = r * np.cos(t) + 10
#     y2 = r * np.sin(t) + 10
#     z2 = t
#     spiral2 = np.column_stack([x2, y2, z2])
    
#     # Combiner les spirales
#     data = np.vstack([spiral1, spiral2])
    
#     # Ajouter des ponts entre les spirales
#     bridge_points_list = []
#     for i in range(n_bridges):
#         idx1 = (i + 1) * len(spiral1) // (n_bridges + 1)
#         idx2 = (i + 1) * len(spiral2) // (n_bridges + 1)
        
#         # Créer des points intermédiaires pour le pont
#         bridge_points = np.linspace(data[idx1], data[idx2], num=20)
#         bridge_points_list.append(bridge_points)
    
#     # Ajouter tous les points des ponts
#     data = np.vstack([data] + bridge_points_list)
    
#     # Ajouter du bruit
#     data += np.random.normal(0, noise, data.shape)
    
#     # Créer des labels pour la visualisation
#     labels = np.arange(len(data)) / len(data)
    
#     assert len(data) == n_points, f"Expected {n_points} points, got {len(data)}"
#     assert len(labels) == n_points, f"Expected {n_points} labels, got {len(labels)}"
    
#     return data, labels

