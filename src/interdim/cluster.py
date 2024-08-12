from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch, MiniBatchKMeans, SpectralClustering, AffinityPropagation, MeanShift, OPTICS
from sklearn.mixture import GaussianMixture

def cluster_data(data, method='dbscan', n_clusters=None, **kwargs):
    """
    Perform clustering on the input data.

    Args:
        data (array-like): Input data to cluster.
        method (str): Clustering method to use. Options: 'kmeans', 'dbscan', 'hdbscan',
                      'agglomerative', 'birch', 'mini_batch_kmeans', 'spectral',
                      'affinity_propagation', 'mean_shift', 'optics', 'gaussian_mixture'.
        n_clusters (int): Number of clusters (for methods that require it).
        **kwargs: Additional arguments to pass to the clustering method.

    Returns:
        array-like: Cluster labels for each data point.
    """
    method = method.lower()
    
    if method == 'kmeans':
        if n_clusters is None:
            raise ValueError("n_clusters must be specified for KMeans clustering")
        clusterer = KMeans(n_clusters=n_clusters, **kwargs)
    elif method == 'dbscan':
        clusterer = DBSCAN(**kwargs)
    elif method == 'hdbscan':
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(**kwargs)
        except ImportError:
            raise ImportError("HDBSCAN is not installed. Install it with 'pip install hdbscan'")
    elif method == 'agglomerative':
        if n_clusters is None:
            raise ValueError("n_clusters must be specified for Agglomerative clustering")
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
    elif method == 'birch':
        clusterer = Birch(n_clusters=n_clusters, **kwargs)
    elif method == 'mini_batch_kmeans':
        if n_clusters is None:
            raise ValueError("n_clusters must be specified for Mini-Batch KMeans clustering")
        clusterer = MiniBatchKMeans(n_clusters=n_clusters, **kwargs)
    elif method == 'spectral':
        if n_clusters is None:
            raise ValueError("n_clusters must be specified for Spectral clustering")
        clusterer = SpectralClustering(n_clusters=n_clusters, **kwargs)
    elif method == 'affinity_propagation':
        clusterer = AffinityPropagation(**kwargs)
    elif method == 'mean_shift':
        clusterer = MeanShift(**kwargs)
    elif method == 'optics':
        clusterer = OPTICS(**kwargs)
    elif method == 'gaussian_mixture':
        if n_clusters is None:
            raise ValueError("n_clusters must be specified for Gaussian Mixture clustering")
        clusterer = GaussianMixture(n_components=n_clusters, **kwargs)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    return clusterer.fit_predict(data)