from sklearn.decomposition import PCA, TruncatedSVD, FastICA, NMF
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

def reduce_dimensionality(data, method='tsne', n_components=2, **kwargs):
    """
    Perform dimensionality reduction on the input data.

    Args:
        data (array-like): Input data to reduce.
        method (str): Reduction method to use. Options: 'pca', 'tsne', 'umap', 'truncated_svd', 
                      'fast_ica', 'nmf', 'isomap', 'lle', 'mds', 'spectral_embedding', 
                      'lda', 'gaussian_random_projection', 'sparse_random_projection'.
        n_components (int): Number of dimensions to reduce to.
        **kwargs: Additional arguments to pass to the reduction method.

    Returns:
        array-like: Reduced data.
    """
    method = method.lower()
    
    if method == 'pca':
        reducer = PCA(n_components=n_components, **kwargs)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, **kwargs)
    elif method == 'umap':
        try:
            from umap import UMAP
            reducer = UMAP(n_components=n_components, **kwargs)
        except ImportError:
            raise ImportError("UMAP is not installed. Install it with 'pip install umap-learn'")
    elif method == 'truncated_svd':
        reducer = TruncatedSVD(n_components=n_components, **kwargs)
    elif method == 'fast_ica':
        reducer = FastICA(n_components=n_components, **kwargs)
    elif method == 'nmf':
        reducer = NMF(n_components=n_components, **kwargs)
    elif method == 'isomap':
        reducer = Isomap(n_components=n_components, **kwargs)
    elif method == 'lle':
        reducer = LocallyLinearEmbedding(n_components=n_components, **kwargs)
    elif method == 'mds':
        reducer = MDS(n_components=n_components, **kwargs)
    elif method == 'spectral_embedding':
        reducer = SpectralEmbedding(n_components=n_components, **kwargs)
    elif method == 'lda':
        reducer = LinearDiscriminantAnalysis(n_components=n_components, **kwargs)
    elif method == 'gaussian_random_projection':
        reducer = GaussianRandomProjection(n_components=n_components, **kwargs)
    elif method == 'sparse_random_projection':
        reducer = SparseRandomProjection(n_components=n_components, **kwargs)
    else:
        raise ValueError(f"Unsupported reduction method: {method}")
    
    return reducer.fit_transform(data)