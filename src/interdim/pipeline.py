import numpy as np
from .reduce import reduce_dimensionality
from .cluster import cluster_data
from .viz import interactive_scatterplot

class InterDimAnalysis:
    def __init__(self, data, reduction_method='tsne', clustering_method='dbscan', 
                 n_components=2, reduction_params=None, clustering_params=None, 
                 viz_func=None, verbose=True):
        self.data = data
        self.reduction_method = reduction_method
        self.clustering_method = clustering_method
        if n_components not in (1,2,3):
            raise ValueError("Argument `n_components` must be 1, 2, or 3.")
        self.n_components = n_components
        self.reduction_params = reduction_params or {}
        self.clustering_params = clustering_params or {}
        self.viz_func = viz_func
        self.verbose = verbose
        
        self.reduced_data = None
        self.cluster_labels = None

    def reduce(self):
        if self.reduction_method is None:
            if self.data.shape[-1] != self.n_components:
                raise ValueError(f"If `clustering_method` is None, the last dimension of data ({self.data.shape[-1]}) must already match n_components ({self.n_components})")
            self.reduced_data = self.data
            if self.verbose:
                print("Skipping dimensionality reduction as reduction_method is None")
        else:
            if self.verbose:
                print(f"Performing dimensionality reduction using {self.reduction_method.upper()} with the following custom params:")
                print(f"\t{self.reduction_params}")
            
            self.reduced_data = reduce_dimensionality(
                self.data, 
                method=self.reduction_method, 
                n_components=self.n_components, 
                **self.reduction_params
            )
        
        if self.verbose:
            print(f"Reduced data shape: {self.reduced_data.shape}")
        
        return self.reduced_data

    def cluster(self):
        if self.clustering_method is None:
            self.cluster_labels = None
            if self.verbose:
                print("Skipping clustering as clustering_method is None")
        else:
            if self.verbose:
                print(f"Performing clustering using {self.clustering_method.upper()} with the following custom params:")
                print(f"\t{self.clustering_params}")
            
            self.cluster_labels = cluster_data(
                self.reduced_data, 
                method=self.clustering_method, 
                **self.clustering_params
            )
            
            if self.verbose:
                n_clusters = len(np.unique(self.cluster_labels))
                print(f"Clustering complete. Number of clusters: {n_clusters}")
        
        return self.cluster_labels

    def visualize(self):
        if self.verbose:
            print("Generating interactive visualization")
        
        vis_data = self.reduced_data if self.reduced_data is not None else self.data
        
        return interactive_scatterplot(
            x=vis_data[:, 0], 
            y=vis_data[:, 1] if self.n_components>1 else None,
            z=vis_data[:, 2] if self.n_components>2 else None,
            interact_fn=self.viz_func,
            marker_color=self.cluster_labels if self.cluster_labels is not None else None,
            marker_size=8,
            marker_opacity=0.7
        )

def analyze_and_visualize(data, reduction_method='tsne', clustering_method='dbscan', 
                          n_components=2, reduction_params=None, clustering_params=None, 
                          viz_func=None, verbose=True):
    """
    Main function for easy use of the InterDim package.

    Parameters:
    - data: numpy array, input data
    - reduction_method: str or None, method for dimensionality reduction (default: 'tsne')
    - clustering_method: str or None, method for clustering (default: 'dbscan')
    - n_components: int, number of components for dimensionality reduction (default: 2)
    - reduction_params: dict, additional parameters for reduction method (optional)
    - clustering_params: dict, additional parameters for clustering method (optional)
    - viz_func: function, custom visualization function (optional)
    - verbose: bool, whether to print progress information (default: True)

    Returns:
    - InterDimAnalysis object
    """
    if verbose:
        print("Starting InterDim analysis pipeline")
    
    analysis = InterDimAnalysis(data, reduction_method, clustering_method, 
                                n_components, reduction_params, clustering_params, 
                                viz_func, verbose)
    analysis.reduce()
    analysis.cluster()
    analysis.visualize()
    
    return analysis