import numpy as np
from sklearn.metrics import silhouette_score, pairwise_distances
from scipy.stats import spearmanr
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import time
from pyDRMetrics.pyDRMetrics import *

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
        # Calculer les k plus proches voisins dans l'espace original
        nn_orig = np.argsort(self.D_high, axis=1)[:, 1:n_neighbors + 1]
        nn_low = np.argsort(self.D_low, axis=1)[:, 1:n_neighbors + 1]
        
        # Calculer le score de préservation
        preservation = np.mean([len(set(nn_orig[i]) & set(nn_low[i])) / n_neighbors 
                            for i in range(len(self.X_high))])
        return round(preservation, 4)
    
    def classification_score(self, n_folds=5):
        if self.labels is None:
            return np.nan
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_low)
        clf = SVC(kernel='linear')
        scores = cross_val_score(clf, X_scaled, self.labels, cv=n_folds)
        return round(scores.mean(), 4)

    def local_distance_preservation_score(self, n_neighbors=10):
        """
        Évalue la préservation des distances locales en comparant les distances
        entre points voisins dans les espaces haute et basse dimension.
        
        Parameters:
        -----------
        n_neighbors : int
            Nombre de voisins à considérer pour l'analyse locale
            
        Returns:
        --------
        float : Score entre 0 et 1 (1 = meilleure préservation)
        """
        try:
            from sklearn.neighbors import NearestNeighbors
            
            # 1. Trouver les k plus proches voisins dans l'espace original
            nbrs_high = NearestNeighbors(n_neighbors=n_neighbors+1).fit(self.X_high)
            high_distances, high_indices = nbrs_high.kneighbors(self.X_high)
            
            # 2. Calculer les distances pour ces mêmes points dans l'espace réduit
            low_distances = np.zeros_like(high_distances)
            for i in range(len(self.X_high)):
                low_distances[i] = np.linalg.norm(
                    self.X_low[high_indices[i]] - self.X_low[i, np.newaxis], 
                    axis=1
                )
            
            # 3. Normaliser les distances dans chaque espace
            high_distances = high_distances / np.max(high_distances)
            low_distances = low_distances / np.max(low_distances)
            
            # 4. Calculer la différence moyenne des distances normalisées
            # Ignorer le premier voisin qui est le point lui-même
            distance_diff = np.abs(high_distances[:, 1:] - low_distances[:, 1:])
            mean_diff = np.mean(distance_diff)
            
            # 5. Convertir en score entre 0 et 1 (1 = meilleure préservation)
            score = 1 - mean_diff
            
            return round(np.clip(score, 0, 1), 4)
        except Exception as e:
            print(f"Erreur dans local_distance_preservation_score: {str(e)}")
            return np.nan
        
    def distance_correlation(self):
        """
        Calcule la corrélation entre les distances des espaces haute et basse dimension.
        Utilise la corrélation de Spearman pour capturer les relations monotones non-linéaires.
        
        Returns:
        --------
        float : Score de corrélation entre 0 et 1 (1 = meilleure préservation)
        """
        try:
            # Extraction des triangles supérieurs des matrices de distance
            # (évite la redondance car les matrices sont symétriques)
            triu_indices = np.triu_indices(self.D_high.shape[0], k=1)
            d_high_flat = self.D_high[triu_indices]
            d_low_flat = self.D_low[triu_indices]
            
            # Normalisation des distances
            d_high_flat = d_high_flat / np.max(d_high_flat)
            d_low_flat = d_low_flat / np.max(d_low_flat)
            
            # Calcul de la corrélation de Spearman
            correlation, _ = spearmanr(d_high_flat, d_low_flat)
            
            # Conversion en score entre 0 et 1
            return round(np.clip(correlation, 0, 1), 4)
        except Exception as e:
            print(f"Erreur dans distance_correlation: {str(e)}")
            return np.nan
    
def evaluate_all_methods(X_original, methods, method_names, labels=None):
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
            'Classification': 'orange',
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
            'Classification': 'red',
            'Time (s)': 'gray'
        }
    }

    for method, name in zip(methods, method_names):
        #try:
        start_time = time.time()
        X_reduced = method(X_original)
        execution_time = time.time() - start_time
            
        evaluator = DimensionalityReductionEvaluator(X_original, X_reduced, method, labels_discrete)

        ##### Réduire les dimensions
        X_reduced = method(X_original)
        
        # Calculer les métriques
        dr_metrics = DRMetrics(X_original, X_reduced)
        
        # Stocker les résultats
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
            'Time (s)': execution_time,
        }
        results.append(metrics)
        #####

    df_results = pd.DataFrame(results).set_index('Method')

    # Apply color coding based on suitability
    def color_code(val, method, metric):
        if method in suitability and metric in suitability[method]:
            color = suitability[method][metric]
            return f'background-color: {color}'
        return ''

    # Apply color coding first, then format the DataFrame
    styled_df = df_results.style.apply(
        lambda x: [color_code(v, x.name, col) for col, v in x.items()], axis=1
    ).format(precision=4)
    return styled_df