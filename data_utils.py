import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
import pickle 
import config


def load_datasets(n_points=1000):
    datasets = {
        'Helicoids': generate_helicoids_dataset(n_points=n_points),
        'Zigzag': generate_zigzag_dataset(n_points=n_points, noise=0.01),
        'Swiss Roll': generate_swiss_roll_dataset(n_points=n_points),
        'Bonhomme': generate_bonhomme_dataset(n_points=n_points)
    }

    # Standardisation
    datasets = {name: data for name, data in datasets.items()}
    return datasets

def load_real_dataset(n_points=None):
    real_data_csv = f"{config.DATA_DIR}/data_real_data.csv"
    df = pd.read_csv(real_data_csv, sep='\t')
    X_processed, y_encoded = preprocess_data_full(df)
    dataset = {'Real Data': (X_processed, y_encoded)}
    return dataset

class DatasetManager:
    def __init__(self, dataset_name, save_path, load_datasets_func):
        """
        Initialize the DatasetManager.

        :param dataset_name: Name of the dataset file (e.g., 'sample_data.pkl').
        :param save_path: Local directory path to save the dataset.
        :param load_datasets_func: Function to load the dataset if not saved.
        """
        self.dataset_name = dataset_name
        self.save_path = save_path
        self.load_datasets_func = load_datasets_func

    def load_dataset(self):
        """
        Load the dataset. Save it locally if it doesn't exist.

        :return: pandas DataFrame containing the dataset.
        """
        full_path = os.path.join(self.save_path, self.dataset_name)
        
        if not self.dataset_exists(full_path):
            print(f"{self.dataset_name} not found locally. Loading using load_datasets...")
            dataset = self.load_datasets_func()
            self.save_dataset(dataset, full_path)
            print(f"Loaded and saved to {full_path}.")
        else:
            print(f"Loading {self.dataset_name} from local storage.")
            dataset = self.load_saved_dataset(full_path)
        
        return dataset

    def dataset_exists(self, path):
        """
        Check if the dataset file exists locally.

        :param path: Full path to the dataset file.
        :return: Boolean indicating existence.
        """
        return os.path.isfile(path)

    def save_dataset(self, dataset, path):
        """
        Save the dataset locally using pickle.

        :param dataset: pandas DataFrame to save.
        :param path: Full path to save the dataset file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(dataset, f)

    def load_saved_dataset(self, path):
        """
        Load the dataset from the saved pickle file.

        :param path: Full path to the dataset file.
        :return: pandas DataFrame.
        """
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset
    
# Générer un Swiss Roll (ruban courbé)
def generate_swiss_roll_dataset(n_points=1000, noise=0):
    t = 2 * np.pi * (1 + np.random.rand(n_points))  # Angle t
    x = t * np.cos(t)  # x-coordinate
    y = 10 * np.random.rand(n_points)  # y-coordinate (épaisseur du ruban)
    z = t * np.sin(t)  # z-coordinate
    data = np.column_stack((x, y, z))
    # Labels pour colorer les points selon t
    labels = t
    return data, labels

# Générer un jeu de données en hélicoïdes imbriquées
def generate_helicoids_dataset(n_points=500, noise=0.17, radius=1.0, pitch=0.5, num_helixes=2):
    data = []
    labels = []
    for i in range(num_helixes):
        t = np.linspace(0, 3 * np.pi, n_points // num_helixes)  # Paramètre angulaire
        x = radius * np.cos(t) + np.random.normal(0, noise, len(t))
        y = radius * np.sin(t) + np.random.normal(0, noise, len(t))
        z = pitch * t + np.random.normal(0, noise, len(t)) + i * pitch * np.pi  # Décalage entre hélicoïdes
        helix = np.column_stack((x, y, z))
        data.append(helix)
        labels.extend([i] * len(t))
    return np.vstack(data), np.array(labels)

# Fonction pour générer des points en forme de bonhomme
def generate_bonhomme(n_points=1000):
    """
    Generates a 3D stick figure composed of head, body, arms raised and legs.
    """
    n_body = n_points // 3    # Points for the body
    n_head = n_points // 6    # Points for the head
    n_limbs = n_points // 6   # Points for each limb (arms and legs)

    # BODY (cylinder along Z-axis)
    z_body = np.random.uniform(-1, 1, n_body)
    theta_body = np.random.uniform(0, 2 * np.pi, n_body)
    x_body = 0.2 * np.cos(theta_body)
    y_body = 0.2 * np.sin(theta_body)
    body = np.column_stack((x_body, y_body, z_body))

    # HEAD (sphere on top of the body)
    theta_head = np.random.uniform(0, 2 * np.pi, n_head)
    phi_head = np.random.uniform(0, np.pi, n_head)
    x_head = 0.3 * np.sin(phi_head) * np.cos(theta_head)
    y_head = 0.3 * np.sin(phi_head) * np.sin(theta_head)
    z_head = 0.3 * np.cos(phi_head) + 1.2  # Position head above body
    head = np.column_stack((x_head, y_head, z_head))

    # Define angle for arms
    angle = np.radians(10)
    cos_ang = np.cos(angle)
    sin_ang = np.sin(angle)

    # ARMS
    # We'll generate arms along a line then rotate them
    # Left Arm: initially horizontal from (-0.5,0,0), then rotated upward.
    theta_arm_left = np.random.uniform(0, 2 * np.pi, n_limbs)
    # Original horizontal arm coordinates before tilt
    x_arm_left_orig = -0.4 + 0.1 * np.cos(theta_arm_left)
    y_arm_left_orig = 0.3 * np.sin(theta_arm_left)
    z_arm_left_orig = np.random.uniform(-0.5, 0.5, n_limbs)
    # Rotate around the Y-axis
    x_arm_left = x_arm_left_orig * cos_ang + z_arm_left_orig * sin_ang
    z_arm_left = -x_arm_left_orig * sin_ang + z_arm_left_orig * cos_ang
    arm_left = np.column_stack((x_arm_left, y_arm_left_orig, z_arm_left))

    # Right Arm: initially horizontal from (0.5,0,0), then rotated upward.
    theta_arm_right = np.random.uniform(0, 2 * np.pi, n_limbs)
    x_arm_right_orig = 0.4 + 0.1 * np.cos(theta_arm_right)
    y_arm_right_orig = 0.3 * np.sin(theta_arm_right)
    z_arm_right_orig = np.random.uniform(-0.5, 0.5, n_limbs)
    # Rotate around the Y-axis°
    x_arm_right = x_arm_right_orig * cos_ang - z_arm_right_orig * sin_ang
    z_arm_right = x_arm_right_orig * sin_ang + z_arm_right_orig * cos_ang
    arm_right = np.column_stack((x_arm_right, y_arm_right_orig, z_arm_right))

    # LEG LEFT (cylinder under body)
    theta_leg_left = np.random.uniform(0, 2 * np.pi, n_limbs)
    z_leg_left = np.random.uniform(-1.5, -1, n_limbs)
    x_leg_left = -0.2 + 0.1 * np.cos(theta_leg_left)
    y_leg_left = 0.1 * np.sin(theta_leg_left)
    leg_left = np.column_stack((x_leg_left, y_leg_left, z_leg_left))

    # LEG RIGHT (cylinder under body)
    theta_leg_right = np.random.uniform(0, 2 * np.pi, n_limbs)
    z_leg_right = np.random.uniform(-1.5, -1, n_limbs)
    x_leg_right = 0.2 + 0.1 * np.cos(theta_leg_right)
    y_leg_right = 0.1 * np.sin(theta_leg_right)
    leg_right = np.column_stack((x_leg_right, y_leg_right, z_leg_right))

    # Combine all body parts
    bonhomme = np.vstack((body, head, arm_left, arm_right, leg_left, leg_right))
    bonhomme = bonhomme[:, [0, 2, 1]]
    return bonhomme

def generate_bonhomme_dataset(n_points=1000):
    """
    Génère un jeu de données en forme de bonhomme 3D.
    
    Parameters:
    -----------
    n_points : int
        Nombre de points à générer.
        
    Returns:
    --------
    data : numpy.ndarray
        Données générées, de forme (n_points, 3).
    labels : numpy.ndarray
        Étiquettes continues entre 0 et 1 basées sur la coordonnée Y.
    """
    data = generate_bonhomme(n_points)
    y_values = data[:, 1]  # Extraire les valeurs Y
    labels = (y_values - y_values.min()) / (y_values.max() - y_values.min())
    return data, labels

def generate_zigzag_dataset(n_points=1000, noise=0.1, random_state=42):
    """
    Génère un jeu de données en forme de zigzag 3D.
    
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
        Étiquettes continues entre 0 et 1 basées sur la progression dans le zigzag.
    """
    np.random.seed(random_state)
    
    # Nombre de segments zigzag
    n_segments = 6
    points_per_segment = n_points // n_segments
    
    points = []
    current_point = np.zeros(3)
    
    # Paires de dimensions pour chaque segment
    dimension_pairs = [(0,1), (1,2), (2,0), (1,0), (2,1)]
    
    for even_dim, odd_dim in dimension_pairs:
        for i in range(points_per_segment):
            t = i / points_per_segment
            new_point = current_point.copy()
            
            # Avance sur la dimension paire
            new_point[even_dim] += t
            
            # Monte puis descend sur la dimension impaire
            if t <= 0.5:
                new_point[odd_dim] += 2 * t
            else:
                new_point[odd_dim] += 2 * (1 - t)
            
            # Ajouter du bruit gaussien
            new_point += np.random.normal(0, noise, size=3)
            points.append(new_point.copy())
            
        current_point = points[-1]
    
    data = np.array(points)
    
    # Créer des labels continus entre 0 et 1 basés sur la progression
    labels = np.linspace(0, 1, len(data))
    
    return data, labels


def preprocess_data_full(df):
    """
    Pré-traite les données en excluant la colonne 'playlist_subgenre', en manipulant les caractéristiques numériques,
    en encodant les caractéristiques catégorielles avec SentenceTransformer, et en normalisant les données.
    
    Args:
        df (pd.DataFrame): DataFrame d'entrée
        scaler (StandardScaler): Scaler pour les caractéristiques numériques
        label_encoder (LabelEncoder): Encodeur pour les labels cibles
        text_encoder (SentenceTransformer): Encodeur textuel utilisé
        is_train (bool): Indique si les données sont d'entraînement

    Retourne:
        X_processed (np.array): Caractéristiques pré-traitées
        y_encoded (np.array): Labels cibles encodés
        scaler (StandardScaler): Scaler utilisé
        label_encoder (LabelEncoder): LabelEncoder utilisé
        text_encoder (SentenceTransformer): Encodeur textuel utilisé
    """

    # Extraire la variable cible
    if 'playlist_genre' in df.columns:
        y = df['playlist_genre']
        df = df.drop(columns=['playlist_genre'])
    elif 'playlist_subgenre' in df.columns:
        y = df['playlist_subgenre']
        df = df.drop(columns=['playlist_subgenre'])
    else:
        raise ValueError("La colonne 'playlist_genre' n'est pas présente dans le dataframe.")

    # Obtenir les caractéristiques numériques
    X_numeric = get_numerical_features(df)

    # Obtenir les caractéristiques catégorielles
    X_categorical = get_categorical_features(df)

    # **Exclure explicitement la colonne 'playlist_subgenre'**
    # (Cela est déjà géré dans get_numerical_features et get_categorical_features)

    # Encodage des autres caractéristiques textuelles avec SentenceTransformer
    # Combiner les colonnes textuelles en une seule chaîne par ligne
    text_data = X_categorical.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    text_embeddings = text_encoder.encode(text_data.tolist())

    # Mise à l'échelle des caractéristiques numériques
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)

    # Préparation de la liste des tableaux de caractéristiques
    feature_arrays = []

    # Ajouter les caractéristiques numériques
    feature_arrays.append(X_numeric_scaled)

    # Ajouter les embeddings textuels si disponibles
    if text_embeddings is not None:
        feature_arrays.append(text_embeddings)

    # Concaténer toutes les caractéristiques
    X_processed = np.hstack(feature_arrays)

    # Encodage des labels cibles
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X_processed, y_encoded

def get_numerical_features(df):
    # Sélectionner uniquement les colonnes numériques
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'playlist_genre' in numerical_cols:
        numerical_cols.remove('playlist_genre')
        numerical_cols.remove('liveness')
        numerical_cols.remove('energy')
        numerical_cols.remove('mode')
    X_numeric = df[numerical_cols]
    return X_numeric

def get_categorical_features(df):
    # Sélectionner uniquement les colonnes catégorielles
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    exclude_cols = ['playlist_genre']
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    X_categorical = df[categorical_cols]
    return X_categorical 

