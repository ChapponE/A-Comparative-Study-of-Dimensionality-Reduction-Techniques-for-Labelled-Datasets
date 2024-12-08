import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import time
from pyDRMetrics.pyDRMetrics import *

from reduction_methods import apply_isomap, apply_lle, apply_mds, apply_tsne

class DimensionalityReductionEvaluator:
    def __init__(self, X_high, X_low, model, labels=None):
        self.X_high = X_high
        self.X_low = X_low
        self.model = model
        self.labels = labels
        self.D_high = pairwise_distances(X_high)
        self.D_low = pairwise_distances(X_low)
    
    def local_distance_preservation(self, n_neighbors=5):
        """Évalue la préservation des distances locales."""
        nn_orig = np.argsort(self.D_high, axis=1)[:, 1:n_neighbors + 1]
        nn_low = np.argsort(self.D_low, axis=1)[:, 1:n_neighbors + 1]

        preservation = np.mean([
            len(set(nn_orig[i]) & set(nn_low[i])) / n_neighbors 
            for i in range(len(self.X_high))
        ])
        return round(preservation, 4)
    
    class EllipticalClassifier:
        """
        Classifieur simple basé sur une distribution gaussienne pour la classe positive.
        """
        def __init__(self):
            self.mean_ = None
            self.cov_inv_ = None
            self.threshold_ = None

        def fit(self, X, y):
            # X: échantillons, y: labels binaires {0,1} avec 1 = classe positive
            X_pos = X[y == 1]
            self.mean_ = np.mean(X_pos, axis=0)
            cov = np.cov(X_pos, rowvar=False)
            self.cov_inv_ = np.linalg.pinv(cov)

            dists = self._mahalanobis(X_pos)
            self.threshold_ = np.max(dists)
            return self

        def predict(self, X):
            dists = self._mahalanobis(X)
            return (dists <= self.threshold_).astype(int)
        
        def _mahalanobis(self, X):
            diff = X - self.mean_
            return np.sum(diff @ self.cov_inv_ * diff, axis=1)


    def classification_score(self):
        """
        Évalue la capacité de discrimination via un classifieur SVC polynomial.
        - Si <= 30 classes distinctes (discret), on entraîne directement sur les classes existantes.
        - Si > 30 classes distinctes (continu), on divise en 30 classes à partir de 31 seuils.
        """
        labels = self.labels
        if labels is None:
            return np.nan

        # Conversion des labels en np.array si nécessaire
        if isinstance(labels, pd.Series):
            labels = labels.values
        elif isinstance(labels, list):
            labels = np.array(labels)

        X_low = self.X_low
        unique_labels = np.unique(labels)
        num_unique = len(unique_labels)

        # Entraînement du classifieur polynomial SVC
        clf = RBFSVCClassifier(C=1.0, gamma='scale')
        clf.fit(X_low, labels)

        # Prédiction et calcul de la précision
        pred_class = clf.predict(X_low)

        if num_unique <= 30:
            # Discret
            # On reconstruit la mapping inverse
            label_mapping_inv = {v: k for k, v in clf.label_mapping.items()}
            true_class = np.array([clf.label_mapping[lab] for lab in labels])
            acc = np.mean(pred_class == true_class)
        else:
            # Continu
            # On refait la même discrétisation pour obtenir la "vraie" classe
            lowest = np.min(labels)
            highest = np.max(labels)
            thresholds = np.linspace(lowest, highest, 31)
            true_class = np.digitize(labels, thresholds[1:-1])
            acc = np.mean(pred_class == true_class)

        return round(acc, 4)
        
    def distance_correlation(self):
        """
        Calcule la corrélation de Spearman entre les distances haute et basse dimension.
        """
        try:
            triu_indices = np.triu_indices(self.D_high.shape[0], k=1)
            d_high_flat = self.D_high[triu_indices]
            d_low_flat = self.D_low[triu_indices]

            d_high_flat = d_high_flat / np.max(d_high_flat)
            d_low_flat = d_low_flat / np.max(d_low_flat)

            correlation, _ = spearmanr(d_high_flat, d_low_flat)
            return round(np.clip(correlation, 0, 1), 4)
        except Exception as e:
            print(f"Erreur dans distance_correlation: {str(e)}")
            return np.nan
    
def evaluate_all_methods(X_original, method_names, labels=None, n_neighbors=None):
    """
    Évalue plusieurs méthodes de réduction de dimension et calcule divers métriques.
    """

    results = []
    
    if labels is not None and np.issubdtype(labels.dtype, np.number):
        is_continuous = len(np.unique(labels)) > 20
        if is_continuous:
            labels_discrete = pd.qcut(labels, q=10, labels=False)
        else:
            labels_discrete = labels
    else:
        labels_discrete = labels

    suitability = {
        'Isomap': {
            'Residual Pearson': 'red',
            'Residual Spearman': 'red',
            'AUC Trustworthiness': 'orange',
            'AUC Continuity': 'orange',
            'Qlocal': 'green',
            'Qglobal': 'green',
            'Distance Correlation': 'green',
            'Classification': 'orange',
            'Time (s)': 'gray'
        },
        'MDS': {
            'Residual Pearson': 'green',
            'Residual Spearman': 'green',
            'AUC Trustworthiness': 'orange',
            'AUC Continuity': 'orange',
            'Qlocal': 'orange',
            'Qglobal': 'green',
            'Distance Correlation': 'green',
            'Classification': 'red',
            'Time (s)': 'gray'
        },
        't-SNE': {
            'Residual Pearson': 'red',
            'Residual Spearman': 'red',
            'AUC Trustworthiness': 'green',
            'AUC Continuity': 'green',
            'Qlocal': 'green',
            'Qglobal': 'orange',
            'Distance Correlation': 'orange',
            'Classification': 'green',
            'Time (s)': 'gray'
        },
        'LLE': {
            'Residual Pearson': 'red',
            'Residual Spearman': 'red',
            'AUC Trustworthiness': 'green',
            'AUC Continuity': 'green',
            'Qlocal': 'green',
            'Qglobal': 'orange',
            'Distance Correlation': 'red',
            'Classification': 'green',
            'Time (s)': 'gray'
        }
    }

    # Gérer n_neighbors
    # Si n_neighbors est un entier, on l'applique à LLE et Isomap.
    # Si c'est un dict, on l'utilise tel quel.
    def get_n_neighbors_for_method(method):
        if isinstance(n_neighbors, dict):
            return n_neighbors.get(method, 10)  # Valeur par défaut si non spécifiée
        else:
            # n_neighbors est un entier, on l'utilise pour LLE et Isomap
            # t-SNE et MDS n'en ont pas besoin
            return n_neighbors if n_neighbors is not None else 10
    
    methods = {}
    for name in method_names:
        if name == 't-SNE':
            methods[name] = lambda x: apply_tsne(x, n_components=2)
        elif name == 'LLE':
            nneigh = get_n_neighbors_for_method('LLE')
            methods[name] = lambda x, nneigh=nneigh: apply_lle(x, n_components=2, n_neighbors=nneigh)
        elif name == 'MDS':
            methods[name] = lambda x: apply_mds(x, n_components=2)
        elif name == 'Isomap':
            nneigh = get_n_neighbors_for_method('Isomap')
            methods[name] = lambda x, nneigh=nneigh: apply_isomap(x, n_components=2, n_neighbors=nneigh)
        else:
            raise ValueError(f"Méthode '{name}' n'est pas supportée.")

    for name in method_names:
        method = methods[name]
        start_time = time.time()
        X_reduced = method(X_original)
        execution_time = time.time() - start_time

        evaluator = DimensionalityReductionEvaluator(X_original, X_reduced, method, labels_discrete)
        dr_metrics = DRMetrics(X_original, X_reduced)

        metrics = {
            'Method': name,
            'Residual Pearson': dr_metrics.Vr,
            'Residual Spearman': dr_metrics.Vrs,
            'AUC Trustworthiness': dr_metrics.AUC_T,
            'AUC Continuity': dr_metrics.AUC_C,
            'Qlocal': dr_metrics.Qlocal,
            'Qglobal': dr_metrics.Qglobal,
            'Distance Correlation': evaluator.distance_correlation(),
            'Classification': evaluator.classification_score(),
            'Time (s)': round(execution_time, 4),
        }
        results.append(metrics)

    df_results = pd.DataFrame(results).set_index('Method')

    def color_code(val, method, metric):
        if method in suitability and metric in suitability[method]:
            color = suitability[method][metric]
            return f'background-color: {color}'
        return ''

    styled_df = df_results.style.apply(
        lambda x: [color_code(v, x.name, col) for col, v in x.items()], axis=1
    ).format(precision=4)
    return styled_df

class RBFSVCClassifier:
    """
    Classifieur basé sur un SVC avec noyau RBF.
    - Pour les labels discrets (<=30 classes distinctes), 
      on utilise toutes les classes telles quelles.
    - Pour les labels continus (>30 classes distinctes),
      on discrétise en 30 classes et on entraîne sur ces classes discrétisées.
    - Utilise des hyperparamètres prédéfinis pour minimiser le surapprentissage.
    """
    def __init__(self, C=1.0, gamma=0.1):
        """
        Parameters:
        - C: Régularisation. Valeur plus faible pour plus de régularisation.
        - gamma: Influence d'un seul point d'entraînement. Valeur plus faible pour une généralisation accrue.
        """
        self.is_discrete = None
        self.label_mapping = None
        self.svc = None
        self.scaler = None
        self.C = C
        self.gamma = gamma

    def fit(self, X, y):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        unique_labels = np.unique(y)
        num_unique = len(unique_labels)

        if num_unique <= 30:
            # Cas discret
            self.is_discrete = True
            # Mapping des labels vers des classes entières
            self.label_mapping = {lab: i for i, lab in enumerate(unique_labels)}
            class_labels = np.array([self.label_mapping[lab] for lab in y])
        else:
            # Cas continu
            self.is_discrete = False
            lowest = np.min(y)
            highest = np.max(y)
            # 31 seuils pour former 30 classes
            thresholds = np.linspace(lowest, highest, 31)
            class_labels = np.digitize(y, thresholds[1:-1])  # classes 0 à 29

        # Entraînement du SVC RBF
        self.svc = SVC(kernel='rbf', C=self.C, gamma=self.gamma, probability=False)
        self.svc.fit(X_scaled, class_labels)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        pred_class = self.svc.predict(X_scaled)
        return pred_class
