# portfolio/allocation.py

import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

def get_hrp_weights(cov_matrix: pd.DataFrame) -> pd.Series:
    """
    Calculates portfolio weights using the Hierarchical Risk Parity (HRP) algorithm.

    Args:
        cov_matrix (pd.DataFrame): A covariance matrix of asset/strategy returns.

    Returns:
        pd.Series: A series of optimal weights, indexed by asset/strategy names.
    """
    if not isinstance(cov_matrix, pd.DataFrame) or cov_matrix.empty:
        return pd.Series()

    corr_matrix = _cov_to_corr(cov_matrix)
    distances = _corr_to_distance(corr_matrix)
    
    # 1. Tree Clustering
    linkage = sch.linkage(squareform(distances), method='single')
    
    # 2. Quasi-Diagonalization (sorts the covariance matrix)
    sorted_indices = sch.to_tree(linkage, rd=False).pre_order()
    sorted_tickers = corr_matrix.index[sorted_indices].tolist()
    
    # 3. Recursive Bisection
    hrp_weights = _get_recursive_bisection(cov_matrix, sorted_tickers)
    
    return pd.Series(hrp_weights, index=cov_matrix.index)

def _cov_to_corr(cov_matrix: pd.DataFrame) -> pd.DataFrame:
    """Converts a covariance matrix to a correlation matrix."""
    std_dev = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(std_dev, std_dev)
    corr_matrix[corr_matrix > 1] = 1 # Correct for floating point errors
    corr_matrix[corr_matrix < -1] = -1
    return corr_matrix

def _corr_to_distance(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """Calculates the distance matrix from a correlation matrix."""
    # Distance is a measure of how "unalike" assets are.
    # d(i, j) = sqrt(0.5 * (1 - corr(i, j)))
    return np.sqrt((1 - corr_matrix) / 2.0)

def _get_cluster_variance(cov_matrix: pd.DataFrame, cluster_assets: list) -> float:
    """Calculates the variance of a cluster of assets."""
    cluster_cov = cov_matrix.loc[cluster_assets, cluster_assets]
    # Inverse-variance weights for assets within the cluster
    ivp = 1. / np.diag(cluster_cov)
    ivp /= ivp.sum()
    cluster_variance = np.dot(ivp.T, np.dot(cluster_cov, ivp))
    return cluster_variance

def _get_recursive_bisection(cov_matrix: pd.DataFrame, sorted_tickers: list) -> pd.Series:
    """
    Performs the recursive bisection allocation.
    """
    weights = pd.Series(1, index=sorted_tickers)
    clusters = [sorted_tickers] # Start with a single cluster of all assets

    while len(clusters) > 0:
        clusters = [c[i:j] for c in clusters for i, j in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        
        for i in range(0, len(clusters), 2):
            cluster1 = clusters[i]
            cluster2 = clusters[i+1]
            
            # Calculate variance for each of the two sub-clusters
            var1 = _get_cluster_variance(cov_matrix, cluster1)
            var2 = _get_cluster_variance(cov_matrix, cluster2)
            
            # Allocation factor based on inverse variance
            alpha = 1 - var1 / (var1 + var2)
            
            # Scale weights in each cluster
            weights[cluster1] *= alpha
            weights[cluster2] *= (1 - alpha)
            
    return weights