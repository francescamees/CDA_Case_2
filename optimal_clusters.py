import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def optimal_k_bic_gmm(X, min_k=1, max_k=10, random_state=None, verbose=True):
    best_k = min_k
    lowest_bic = np.inf

    for k in range(min_k, max_k + 1):
        gmm = GaussianMixture(n_components=k, random_state=random_state).fit(X)
        bic = gmm.bic(X)
        if bic < lowest_bic:
            lowest_bic = bic
            best_k = k

    if verbose:
        print(f"GMM-BIC: optimal K = {best_k} (BIC = {lowest_bic:.2f})")

    return best_k

def optimal_k_silhouette(X, min_k=2, max_k=10, random_state=None, verbose=True):
    best_k = min_k
    best_score = -1

    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state).fit(X)
        labels = kmeans.labels_
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k

    if verbose:
        print(f"Silhouette: optimal K = {best_k:.0f} (score = {best_score:.3f})")

    return best_k


def gap_statistic_optimal_k(X, clustersNr=10, Nsim=20, random_state=None, verbose=True):
    """
    Computes the optimal number of clusters using the Gap Statistic method.
    
    Parameters:
    - X: np.ndarray, shape (n_samples, n_features), the input data.
    - clustersNr: int, the maximum number of clusters to try.
    - Nsim: int, number of uniform reference simulations.
    - random_state: for reproducibility.
    - verbose: bool, whether to print the selected K.

    Returns:
    - K_opt: int, the estimated optimal number of clusters.
    """
    N, p = X.shape
    minX = np.min(X, axis=0)
    maxX = np.max(X, axis=0)
    list_of_clusters = range(1, clustersNr + 1)

    W = np.zeros(clustersNr) # within-cluster dispersion
    Wu = np.zeros((clustersNr, Nsim)) # within-cluster dispersion for uniform data

    rng = np.random.default_rng(random_state)

    for k in list_of_clusters:
        # Real data clustering
        kmeans = KMeans(n_clusters=k, random_state=random_state).fit(X)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        for c in range(k):
            idx = labels == c
            W[k - 1] += np.sum((X[idx] - centers[c])**2)

        # Simulations
        for j in range(Nsim):
            Xu = rng.uniform(minX, maxX, size=(N, p))
            kmeansU = KMeans(n_clusters=k, random_state=random_state).fit(Xu)
            centers_u = kmeansU.cluster_centers_
            labels_u = kmeansU.labels_

            for c in range(k):
                idx_u = labels_u == c
                Wu[k - 1, j] += np.sum((Xu[idx_u] - centers_u[c])**2)

    # Compute gap statistics
    logW = np.log(W)
    logWu = np.log(Wu)
    Gk = np.mean(logWu, axis=1) - logW
    sk = np.std(logWu, axis=1) * np.sqrt(1 + 1.0 / Nsim)

    # Apply the 1-standard deviation rule (ESL Eq. 14.39)
    K_opt_idx = np.where(Gk[:-1] >= Gk[1:] - sk[1:])[0]
    if K_opt_idx.size == 0:
        K_opt = clustersNr
    else:
        K_opt = list_of_clusters[K_opt_idx[0]]

    if verbose:
        print(f"Gap-statistic: optimal K = {K_opt}")

    return K_opt