import numpy as np
from .reduce import reduce_dimensionality
from .cluster import cluster_data
from .score import score_clustering
from .viz import interactive_scatterplot

class InterDimAnalysis:
    def __init__(self, data, true_labels=None, viz_func=None, verbose=True):
        self.data = data
        self.true_labels = true_labels
        self.viz_func = viz_func
        self.verbose = verbose
        
        self.reduced_data = None
        self.cluster_labels = None
        self.score_value = None

    def reduce(self, method='tsne', n_components=2, **kwargs):
        if method is None:
            if self.data.shape[-1] != n_components:
                raise ValueError(f"If method is None, the last dimension of data ({self.data.shape[-1]}) must already match n_components ({n_components})")
            self.reduced_data = self.data
            if self.verbose:
                print("Skipping dimensionality reduction as method is None")
        else:
            if self.verbose:
                print(f"Performing dimensionality reduction using {method.upper()} with the following custom params:")
                print(f"\t{kwargs}")
            
            self.reduced_data = reduce_dimensionality(
                self.data, 
                method=method, 
                n_components=n_components, 
                **kwargs
            )
        
        if self.verbose:
            print(f"Reduced data shape: {self.reduced_data.shape}")
        
        return self.reduced_data

    def cluster(self, method='dbscan', **kwargs):
        if method is None:
            self.cluster_labels = None
            if self.verbose:
                print("Skipping clustering as method is None")
        else:
            if self.verbose:
                print(f"Performing clustering using {method.upper()} with the following custom params:")
                print(f"\t{kwargs}")
            
            self.cluster_labels = cluster_data(
                self.reduced_data, 
                method=method, 
                **kwargs
            )
            
            if self.verbose:
                n_clusters = len(np.unique(self.cluster_labels))
                print(f"Clustering complete. Number of clusters: {n_clusters}")
        
        return self.cluster_labels

    def score(self, method='silhouette', true_labels=None):
        if self.reduced_data is None:
            raise ValueError("You must run the 'reduce' method before scoring.")
        if self.cluster_labels is None:
            raise ValueError("You must run the 'cluster' method before scoring.")
        
        # Use true_labels passed to this method if provided, otherwise use the ones from initialization
        labels_to_use = true_labels if true_labels is not None else self.true_labels
        
        self.score_value = score_clustering(self.reduced_data, self.cluster_labels, labels_to_use, method)
        
        if self.verbose:
            print(f"Clustering Evaluation Result ({method}):")
            print(f"Score: {self.score_value}")
        
        return self.score_value


    def visualize(self, n_components=None):
        if self.verbose:
            print("Generating interactive visualization")
        
        vis_data = self.reduced_data if self.reduced_data is not None else self.data
        
        if n_components is None:
            n_components = vis_data.shape[1]
        
        return interactive_scatterplot(
            x=vis_data[:, 0], 
            y=vis_data[:, 1] if n_components > 1 else None,
            z=vis_data[:, 2] if n_components > 2 else None,
            interact_fn=self.viz_func,
            marker_color=self.cluster_labels if self.cluster_labels is not None else None,
            marker_size=8,
            marker_opacity=0.7
        )

def analyze_and_visualize(data, n_components=2, true_labels=None, 
                          reduction_method='tsne', clustering_method='dbscan', scoring_method='silhouette', 
                          reduction_params=None, clustering_params=None, 
                          viz_func=None, verbose=True):
    """
    Main function for easy use of the InterDim package.

    Parameters:
    - data: numpy array, input data
    - n_components: int, number of components for dimensionality reduction (default: 2)
    - true_labels: array-like, true labels for supervised scoring methods (optional)
    - reduction_method: str or None, method for dimensionality reduction (default: 'tsne')
    - clustering_method: str or None, method for clustering (default: 'dbscan')
    - scoring_method: str, method for scoring clustering performance (default: 'silhouette')
    - reduction_params: dict, additional parameters for reduction method (optional)
    - clustering_params: dict, additional parameters for clustering method (optional)
    - viz_func: function, custom visualization function (optional)
    - verbose: bool, whether to print progress information (default: True)

    Returns:
    - InterDimAnalysis object
    """
    
    analysis = InterDimAnalysis(data, viz_func, verbose, true_labels)
    analysis.reduce(method=reduction_method, n_components=n_components, **(reduction_params or {}))
    analysis.cluster(method=clustering_method, **(clustering_params or {}))
    analysis.score(method=scoring_method)
    analysis.visualize(n_components=n_components)
    
    return analysis