from typing import Optional, Callable, Union
import numpy as np
from .reduce import reduce_dimensionality, ReductionMethodType
from .cluster import cluster_data, ClusteringMethodType
from .score import score_clustering, ScoreMethodType
from .viz import interactive_scatterplot

class InterDimAnalysis:
    def __init__(
        self,
        data: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
        viz_func: Optional[Callable] = None,
        verbose: bool = True
    ):
        """
        Initialize the InterDimAnalysis object.

        Args:
            data: Input data for analysis.
            true_labels: True labels for supervised evaluation (optional).
            viz_func: Custom visualization function (optional).
            verbose: Whether to print progress information.
        """
        self.data = data
        self.true_labels = true_labels
        self.viz_func = viz_func
        self.verbose = verbose
        
        self.reduced_data: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.score_value: Optional[Union[float, np.ndarray]] = None

    def reduce(
        self,
        method: ReductionMethodType = 'tsne',
        n_components: int = 2,
        **kwargs
    ) -> np.ndarray:
        """
        Perform dimensionality reduction on the data.

        Args:
            method: Dimensionality reduction method to use.
            n_components: Number of components to reduce to.
            **kwargs: Additional arguments for the reduction method.

        Returns:
            The reduced data.
        """
        if method is None:
            if self.data.shape[1] != n_components:
                raise ValueError(f"If method is None, the number of features ({self.data.shape[1]}) must match n_components ({n_components})")
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

    def cluster(
        self,
        method: ClusteringMethodType = 'dbscan',
        **kwargs
    ) -> np.ndarray:
        """
        Perform clustering on the reduced data.

        Args:
            method: Clustering method to use.
            **kwargs: Additional arguments for the clustering method.

        Returns:
            The cluster labels.

        Raises:
            ValueError: If dimensionality reduction hasn't been performed yet.
        """
        if self.reduced_data is None:
            raise ValueError("You must run the 'reduce' method before clustering.")

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

    def score(
        self,
        method: ScoreMethodType = 'silhouette',
        true_labels: Optional[np.ndarray] = None
    ) -> Union[float, np.ndarray]:
        """
        Evaluate the clustering performance.

        Args:
            method: Scoring method to use.
            true_labels: True labels for supervised scoring methods (optional).

        Returns:
            The computed score.

        Raises:
            ValueError: If clustering hasn't been performed yet.
        """
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

    def visualize(self, n_components: Optional[int] = None):
        """
        Generate an interactive visualization of the reduced data.

        Args:
            n_components: Number of components to visualize (2 or 3).

        Returns:
            A Dash application instance for the interactive plot.

        Raises:
            ValueError: If dimensionality reduction hasn't been performed yet.
        """
        if self.reduced_data is None:
            raise ValueError("You must run the 'reduce' method before visualizing.")

        if self.verbose:
            print("Generating interactive visualization")
        
        vis_data = self.reduced_data
        
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


def analyze_and_visualize(
    data: np.ndarray,
    n_components: int = 2,
    true_labels: Optional[np.ndarray] = None, 
    reduction_method: ReductionMethodType = 'tsne',
    clustering_method: ClusteringMethodType = 'dbscan',
    scoring_method: ScoreMethodType = 'silhouette', 
    reduction_params: Optional[dict] = None,
    clustering_params: Optional[dict] = None, 
    viz_func: Optional[Callable] = None,
    verbose: bool = True
) -> InterDimAnalysis:
    """
    Main function for easy use of the InterDim package.

    Args:
        data: Input data for analysis.
        n_components: Number of components for dimensionality reduction.
        true_labels: True labels for supervised scoring methods (optional).
        reduction_method: Method for dimensionality reduction.
        clustering_method: Method for clustering.
        scoring_method: Method for scoring clustering performance.
        reduction_params: Additional parameters for reduction method (optional).
        clustering_params: Additional parameters for clustering method (optional).
        viz_func: Custom visualization function (optional).
        verbose: Whether to print progress information.

    Returns:
        An InterDimAnalysis object with the analysis results.
    """
    analysis = InterDimAnalysis(data, true_labels, viz_func, verbose)
    analysis.reduce(method=reduction_method, n_components=n_components, **(reduction_params or {}))
    analysis.cluster(method=clustering_method, **(clustering_params or {}))
    analysis.score(method=scoring_method)
    analysis.visualize(n_components=n_components)
    
    return analysis