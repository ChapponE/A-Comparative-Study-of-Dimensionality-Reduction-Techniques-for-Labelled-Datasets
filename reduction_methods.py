import numpy as np
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.manifold import LocallyLinearEmbedding

# Les fonctions pour les différentes méthodes de réduction de dimension
def apply_tsne(data, n_components=2):
    tsne = manifold.TSNE(n_components=n_components, random_state=42)
    return tsne.fit_transform(data)

def apply_lle(data, n_components=2, n_neighbors=10):
    if hasattr(data, 'to_numpy'):
        data = data.to_numpy()
    lle = CustomLLE(n_neighbors=n_neighbors, n_components=n_components)
    lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, 
                                        n_components=n_components,
                                        random_state=42)
    return lle.fit_transform(data)

def apply_isomap(data, n_components=2):
    isomap = CustomIsomap(n_neighbors=10, n_components=n_components)
    return isomap.fit_transform(data)

def apply_mds(data, n_components=2):
    mds = CustomMDS(n_components=n_components)
    return mds.fit_transform(data)

# Custom LLE
class CustomLLE:
    def __init__(self, n_neighbors=15, n_components=2):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.weights = None

    def _find_neighbors(self, X):
        """Find the k-nearest neighbors for each point."""
        n_samples = X.shape[0]
        distances = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
        indices = np.argsort(distances, axis=1)[:, 1:self.n_neighbors + 1]
        return indices.astype(np.int32)

    def _compute_weights(self, X, neighbors):
        """Compute reconstruction weights for each point."""
        n_samples = X.shape[0]
        weights = np.zeros((n_samples, n_samples), dtype=np.float64)
        
        for i in range(n_samples):
            z = X[neighbors[i]] - X[i]
            C = np.dot(z, z.T).astype(np.float64)
            reg = np.eye(self.n_neighbors, dtype=np.float64) * 1e-3 * np.trace(C)
            C += reg
            w = np.linalg.solve(C, np.ones(self.n_neighbors, dtype=np.float64))
            w /= np.sum(w)
            weights[i, neighbors[i]] = w
            
        return weights

    def fit_transform(self, X):
        """Apply LLE to the data X."""
        X = np.asarray(X, dtype=np.float64)
        neighbors = self._find_neighbors(X)
        self.weights = self._compute_weights(X, neighbors)
        return self._compute_embedding(self.weights)

    def _compute_embedding(self, weights):
        """Compute the final embedding."""
        n_samples = weights.shape[0]
        M = np.eye(n_samples, dtype=np.float64) - weights
        M = np.dot(M.T, M)
        M.flat[::n_samples + 1] += 1e-7
        eigenvals, eigenvecs = np.linalg.eigh(M)
        index = np.argsort(eigenvals)[1:self.n_components + 1]
        eigenvals = eigenvals[index]
        embedding = eigenvecs[:, index] * np.sqrt(eigenvals)
        return embedding

    def local_distance_correlation(self, X, X_reduced, window_size=10):
        """
        Calculate the local distance correlation between the original and reduced data using a sliding window approach.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Original high-dimensional data.
        X_reduced : array-like, shape (n_samples, n_components)
            Reduced low-dimensional data.
        window_size : int, optional (default=10)
            Size of the sliding window for local correlation computation.

        Returns:
        --------
        float : Mean local distance correlation score between 0 and 1 (1 = best preservation)
        """
        from scipy.spatial import distance  # Assurez-vous que cette importation est présente
        import numpy as np

        try:
            n = X.shape[0]
            # Calculer les matrices de distance
            dx = distance.squareform(distance.pdist(X))
            dy = distance.squareform(distance.pdist(X_reduced))
            ldc_values = np.zeros(n)

            for i in range(n):
                # Définir les indices de la fenêtre glissante
                start = max(0, i - window_size)
                end = min(n, i + window_size + 1)
                # Extraire les sous-matrices locales
                dx_window = dx[start:end, start:end].ravel()
                dy_window = dy[start:end, start:end].ravel()
                # Calculer la corrélation de Pearson entre les distances locales
                if np.std(dx_window) == 0 or np.std(dy_window) == 0:
                    r = 0  # Éviter la division par zéro
                else:
                    r = np.corrcoef(dx_window, dy_window)[0, 1]
                ldc_values[i] = r

            # Calculer la moyenne des scores LDC, en ignorant les NaN éventuels
            mean_ldc = np.nanmean(ldc_values)
            # Normaliser le score entre 0 et 1
            return round(np.clip(mean_ldc, 0, 1), 4)

        except Exception as e:
            print(f"Erreur dans local_distance_correlation: {str(e)}")
            return np.nan



# Implementation de MDS

class CustomMDS:
    def __init__(self, n_components=2, max_iter=300, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        
    def _compute_distances(self, X):
        """Calcule la matrice des distances euclidiennes."""
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = np.sqrt(np.sum((X[i] - X[j])**2))
                distances[i,j] = distances[j,i] = dist
                
        return distances
    
    def _stress(self, distances_original, distances_embedded):
        """Calcule le stress de Kruskal."""
        numerator = np.sum((distances_embedded - distances_original)**2)
        denominator = np.sum(distances_embedded**2)
        return np.sqrt(numerator / denominator)
    
    def fit_transform(self, X):
        """
        Applique MDS aux données X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Les données à transformer
            
        Returns:
        --------
        X_embedded : array-like, shape (n_samples, n_components)
            Les données transformées
        """
        n_samples = X.shape[0]
        
        # Calculer les distances originales
        distances_original = self._compute_distances(X)
        
        # Initialiser les coordonnées avec PCA pour une meilleure convergence
        pca = PCA(n_components=self.n_components)
        embedding = pca.fit_transform(X)
        
        # Optimisation itérative
        stress_old = float('inf')
        
        for iteration in range(self.max_iter):
            # Calculer les distances dans l'espace réduit
            distances_embedded = self._compute_distances(embedding)
            
            # Calculer le stress
            stress = self._stress(distances_original, distances_embedded)
            
            # Vérifier la convergence
            if abs(stress - stress_old) < self.tol:
                break
                
            # Mettre à jour les coordonnées
            B = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    if i != j and distances_embedded[i,j] > 1e-8:
                        B[i,j] = -distances_original[i,j] / distances_embedded[i,j]
                B[i,i] = -np.sum(B[i,:])
            
            # Mettre à jour l'embedding
            embedding = np.dot(B, embedding) / n_samples
            
            stress_old = stress
        
        return embedding

class CustomIsomap:
    def __init__(self, n_neighbors=10, n_components=2):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        
    def _compute_knn_graph(self, X):
        """Calcule le graphe des k plus proches voisins."""
        n_samples = X.shape[0]
        
        # Calcul des distances euclidiennes entre tous les points
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = np.sqrt(np.sum((X[i] - X[j])**2))
                distances[i,j] = distances[j,i] = dist
        
        # Pour chaque point, trouve les k plus proches voisins
        neighbors = np.zeros((n_samples, self.n_neighbors), dtype=int)
        for i in range(n_samples):
            neighbors[i] = np.argsort(distances[i])[1:self.n_neighbors+1]
            
        return distances, neighbors
    
    def _compute_geodesic_distances(self, distances, neighbors):
        """Calcule les distances géodésiques en utilisant l'algorithme de Floyd-Warshall."""
        n_samples = distances.shape[0]
        geodesic = np.full((n_samples, n_samples), np.inf)
        
        # Initialiser avec les distances directes pour les voisins
        for i in range(n_samples):
            geodesic[i,i] = 0
            for j in neighbors[i]:
                geodesic[i,j] = geodesic[j,i] = distances[i,j]
        
        # Floyd-Warshall
        for k in range(n_samples):
            for i in range(n_samples):
                for j in range(n_samples):
                    if geodesic[i,k] + geodesic[k,j] < geodesic[i,j]:
                        geodesic[i,j] = geodesic[i,k] + geodesic[k,j]
        
        # Remplacer les inf restants par la distance max
        max_dist = np.max(geodesic[~np.isinf(geodesic)])
        geodesic[np.isinf(geodesic)] = max_dist
        
        return geodesic
    
    def _mds_embedding(self, geodesic_distances):
        """Applique MDS classique aux distances géodésiques en utilisant CustomMDS."""
        mds = CustomMDS(n_components=self.n_components)
        embedding = mds.fit_transform(geodesic_distances)
        return embedding
    
    def fit_transform(self, X):
        """
        Applique Isomap aux données X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Les données à transformer
            
        Returns:
        --------
        X_embedded : array-like, shape (n_samples, n_components)
            Les données transformées
        """
        # Calcul du graphe des k plus proches voisins
        distances, neighbors = self._compute_knn_graph(X)
        
        # Calcul des distances géodésiques
        geodesic_distances = self._compute_geodesic_distances(distances, neighbors)
        
        # Application de MDS aux distances géodésiques
        embedding = self._mds_embedding(geodesic_distances)
        
        return embedding

    def geodesic_error(self, X, X_reduced):
        """
        Calculate the geodesic error between the original data and the reduced data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The original high-dimensional data.
        
        X_reduced : array-like, shape (n_samples, n_components)
            The reduced low-dimensional data.
        
        Returns:
        --------
        error : float
            The mean geodesic error.
        """
        # Compute the geodesic distances in the original space
        distances, neighbors = self._compute_knn_graph(X)
        geodesic_distances = self._compute_geodesic_distances(distances, neighbors)
        
        # Compute the Euclidean distances in the reduced space
        n_samples = X_reduced.shape[0]
        reduced_distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = np.sqrt(np.sum((X_reduced[i] - X_reduced[j])**2))
                reduced_distances[i, j] = reduced_distances[j, i] = dist
        
        # Calculate the geodesic error
        error = np.mean(np.abs(geodesic_distances - reduced_distances))
        return error

