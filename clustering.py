import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import pandas as pd
from sklearn.metrics import silhouette_score
from utils import FEATURES, MODEL_KWARGS
from tqdm import tqdm

class Clustering(object):
    """ General clustering class """
    
    def __init__(self, type: str, data: pd.DataFrame, features: list[str], **kwargs):
        assert type in ['kmeans', 'hierarchical', 'gmm'], "Invalid clustering type"
        self.type = type
        self.data = data
        self.X = data[features].values
        self.kwargs = kwargs
        self.model = self._get_model()
        self._is_fitted = False

    def _get_model(self):
        """ Get the clustering model based on the type """
        if self.type == 'kmeans':
            return KMeans(**self.kwargs)
        elif self.type == 'hierarchical':
            return AgglomerativeClustering(**self.kwargs)
        elif self.type == 'gmm':
            return GaussianMixture(**self.kwargs)
        
    def fit(self):
        """ Fit the clustering model to the data """
        assert hasattr(self.model, 'fit'), "Model does not have a fit method"
        self.model.fit(self.X)
        self._is_fitted = True
        return self
    
    def get_labels(self) -> np.ndarray:
        """ Append an extra column to the data with the cluster labels 
            and return the labels """
        if not self._is_fitted:
            self.fit()
        assert hasattr(self.model, 'labels_') or hasattr(self.model, 'predict'), "Model does not have labels_ or predict method"
        if hasattr(self.model, 'labels_'):
            self.data['cluster_idx'] = self.model.labels_
        else:
            self.data['cluster_idx'] = self.model.predict(self.X)
        return self.data['cluster_idx'].values
    
    def get_optimal_k(self, k_min: int = 0, k_max: int = 10,
                        method: str = 'silhouette', n_sim: int = 20,
                        verbose: bool = True) -> tuple[int, dict]:
            """ Get the optimal number of clusters """
            assert k_min > 0 and k_max > k_min, "Invalid range for k"
            if method == 'silhouette':
                return self._optimal_k_silhouette(k_min, k_max, verbose)
            elif method == 'bic':
                return self._optimal_k_bic_gmm(k_min, k_max, verbose)
            elif method == 'gap':
                return self._optimal_k_gap_statistic(k_min, k_max, n_sim, verbose)
            else:
                raise ValueError("Invalid method for optimal K")
            
    def get_inertia_for_k_range(self, k_min: int = 2, k_max: int = 10, verbose: bool = True) -> dict:
        """ Compute the sum of squared distances (inertia) for different k (KMeans only) """
        assert self.type == 'kmeans', "Inertia is only defined for KMeans"
        inertias = {}
        pbar = tqdm(range(k_min, k_max + 1), desc="Inertia vs K", disable=not verbose)
        
        for k in pbar:
            self._set_n_clusters(k)
            self.fit()
            inertias[k] = self.model.inertia_
            if verbose:
                pbar.set_postfix({"Inertia": inertias[k]})
        
        return inertias

    def feature_importance_via_wcss(self, verbose=False):
        """
        Estimate feature importance using a cluster-based weighting technique.
        Measures the change in WCSS (within-cluster sum of squares) when each feature is removed.
        The more WCSS increases upon feature removal, the more important the feature is.
        """
        assert self.type == 'kmeans', "Feature importance via WCSS is only supported for KMeans"

        if hasattr(self.X[0], 'dtype') and self.X[0].dtype.names is not None:
            feature_names = self.X[0].dtype.names
        else:
            feature_names = self.data.columns

        original_features = self.data.columns.intersection(feature_names).tolist()

        # Fit model with all features to get baseline WCSS
        self.fit()
        baseline_wcss = self.model.inertia_
        n_clusters = self.model.n_clusters
        random_state = getattr(self.model, 'random_state', None)


        if verbose:
            print(f"Baseline WCSS (all features): {baseline_wcss:.2f}")

        importances = {}

        for feature in original_features:
            if verbose:
                print(f"\nEvaluating feature: {feature}")

            # Drop the feature
            reduced_data = self.data.drop(columns=[feature])

            # Fit new model
            model = KMeans(n_clusters=n_clusters, random_state=random_state)
            model.fit(reduced_data)

            reduced_wcss = model.inertia_

            # The increase in WCSS indicates importance
            delta_wcss = reduced_wcss - baseline_wcss
            importances[feature] = delta_wcss

            if verbose:
                print(f"WCSS without {feature}: {reduced_wcss:.2f} (Δ = {delta_wcss:.2f})")

        # Normalize importances
        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}
        else:
            importances = {k: 0.0 for k in importances}

        return importances


        

    
    def _set_n_clusters(self, n_clusters: int):
        """ Set the number of clusters for the model """
        if self.type == 'kmeans':
            self.kwargs['n_clusters'] = n_clusters
        elif self.type == 'hierarchical':
            self.kwargs['n_clusters'] = n_clusters
        elif self.type == 'gmm':
            self.kwargs['n_components'] = n_clusters
        else:
            raise ValueError("Invalid clustering type")
        self.model = self._get_model()
        if 'cluster_idx' in self.data.columns:
            del self.data['cluster_idx']
        self._is_fitted = False
        return self
    
    def _optimal_k_silhouette(self, k_min: int, k_max: int,
                            verbose: bool = True) -> tuple[int, dict]:
        """ Find the optimal number of clusters using silhouette score """
        assert self.type in ['kmeans', 'hierarchical'], "Silhouette is only applicable for KMeans and Hierarchical"
        best_k = k_min
        best_score = -1
        scores = {}

        pbar = tqdm(range(k_min, k_max + 1), desc=f"Finding optimal K for {self.type.capitalize()}", unit="k")
        
        for k in pbar if verbose else range(k_min, k_max + 1):
            self._set_n_clusters(k)
            self.fit()
            labels = self.get_labels()
            score = silhouette_score(self.X, labels)
            scores[k] = score
            if score > best_score:
                best_score = score
                best_k = k
            if verbose:
                pbar.set_postfix({"K": k})
        if verbose:
            print(f"Silhouette: optimal K = {best_k:.0f} (score = {best_score:.3f})")
        return best_k, scores
    
    def _optimal_k_bic_gmm(self, k_min: int, k_max: int,
                          verbose: bool = True) -> tuple[int, dict]:
        """ Find the optimal number of clusters using BIC for GMM """
        assert self.type == 'gmm', "BIC is only applicable for GMM"
        best_k = k_min
        lowest_bic = np.inf
        bics = {}
        pbar = tqdm(range(k_min, k_max + 1), desc=f"Finding optimal K for {self.type.capitalize()}", unit="k")
        for k in pbar if verbose else range(k_min, k_max + 1):
            self._set_n_clusters(k)
            self.fit()
            bic = self.model.bic(self.X)
            bics[k] = bic
            if bic < lowest_bic:
                lowest_bic = bic
                best_k = k
            if verbose:
                pbar.set_postfix({"K": k})
        if verbose:
            print(f"GMM-BIC: optimal K = {best_k} (BIC = {lowest_bic:.2f})")
        return best_k, bics
    
    def _optimal_k_gap_statistic(self, k_min: int, k_max: int,
                                n_sim: int = 20,
                                verbose: bool = True) -> tuple[int, dict]:
        """ Find the optimal number of clusters using Gap Statistic
            NOTE: Code from exercise week 9)
        """
        assert self.type == 'kmeans', "Gap statistic is only applicable for KMeans"
        N, p = self.X.shape
        minX = np.min(self.X, axis=0)
        maxX = np.max(self.X, axis=0)
        list_of_clusters = range(k_min, k_max + 1)
        W = np.zeros(k_max-k_min+1)
        Wu = np.zeros((k_max-k_min+1, n_sim))
        rng = np.random.default_rng()
        pbar = tqdm(list_of_clusters, desc=f"Finding optimal K for {self.type.capitalize()}", unit="k")
        for k in pbar if verbose else list_of_clusters:
            self._set_n_clusters(k)
            self.fit()
            labels = self.get_labels()
            centers = self.model.cluster_centers_

            for c in range(k):
                idx = labels == c
                W[k - k_min] += np.sum((self.X[idx] - centers[c])**2)

            # Simulations
            for j in range(n_sim):
                Xu = rng.uniform(minX, maxX, size=(N, p))
                kmeansU = KMeans(n_clusters=k, random_state=0).fit(Xu)
                centers_u = kmeansU.cluster_centers_
                labels_u = kmeansU.labels_

                for c in range(k):
                    idx_u = labels_u == c
                    Wu[k - k_min, j] += np.sum((Xu[idx_u] - centers_u[c])**2)
            if verbose:
                pbar.set_postfix({"K": k})
        # Compute gap statistics
        logW = np.log(W)
        logWu = np.log(Wu)
        Gk = np.mean(logWu, axis=1) - logW
        sk = np.std(logWu, axis=1) * np.sqrt(1 + 1.0 / 20)
        gap_scores = dict(zip(list_of_clusters, Gk))

        # Apply the 1-standard deviation rule (ESL Eq. 14.39)
        K_opt_idx = np.where(Gk[:-1] >= Gk[1:] - sk[1:])[0]
        if K_opt_idx.size == 0:
            K_opt = k_max
        else:
            K_opt = list_of_clusters[K_opt_idx[0]]
        if verbose:
            print(f"Gap-statistic: optimal K = {K_opt}")
        return K_opt, gap_scores

if __name__ == '__main__':
    # Example usage for optimal K
    data = pd.read_csv('data/data_pre_processed.csv')
    for model_type in ['kmeans', 'hierarchical', 'gmm']:
        if model_type == 'gmm':
            method = 'bic'
        elif model_type == 'hierarchical':
            method = 'silhouette'
        else:
            method = 'gap'
        
        model = Clustering(model_type, data, FEATURES, **MODEL_KWARGS[model_type])
        optimal_k, scores = model.get_optimal_k(k_min=2, k_max=20, method=method, verbose=True)