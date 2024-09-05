from typing import Literal, Optional, Callable, Union
import numpy as np
import warnings
import dash
from .reduce import reduce_dimensionality, ReductionMethodType
from .cluster import cluster_data, ClusteringMethodType
from .score import score_clustering, ScoreMethodType
from .viz import interactive_scatterplot

class InterDimAnalysis:
    def __init__(
        self,
        data: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
        verbose: bool = True
    ):
        """
        Initialize the InterDimAnalysis object.

        Args:
            data: Input data for analysis.
            true_labels: True labels for supervised evaluation (optional).
            verbose: Whether to print progress information.
        """
        self.data = data
        self.true_labels = true_labels
        self.verbose = verbose
        
        self.reduced_data: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_score: Optional[Union[float, np.ndarray]] = None

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
                if len(kwargs) == 0:
                    print(f"Performing dimensionality reduction via {method.upper()} with default arguments.")
                else:
                    print(f"Performing dimensionality reduction via {method.upper()} with the following custom arguments:")
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
                if len(kwargs) == 0:
                    print(f"Performing clustering via {method.upper()} with default arguments.")
                else:
                    print(f"Performing clustering via {method.upper()} with the following custom arguments:")
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
        
        self.cluster_score = score_clustering(self.reduced_data, self.cluster_labels, labels_to_use, method)
        
        if self.verbose:
            print(f"Clustering Evaluation Result ({method}):")
            print(f"Score: {self.cluster_score}")
        
        return self.cluster_score

    def show(
            self, 
            n_components: Optional[int] = 3, 
            which_data: str = 'reduced', 
            interact_fn: Optional[Callable] = None,
            x_label: Optional[str] = None,
            y_label: Optional[str] = None,
            z_label: Optional[str] = None,
            marker_color: Optional[Union[str, np.ndarray]] = None,
            marker_size: int = 5,
            marker_opacity: float = 0.5,
            interact_mode: Literal["hover", "click"] = 'hover',
            run_server: bool = True
        ) -> dash.Dash:
        """
        Generate an interactive visualization of the data.

        Args:
            n_components: Number of components to show (1, 2, or 3).
            which_data: Which dataset to show ('original' or 'reduced').
            interact_fn: Function to call on interaction events.
            x_label: Label for X-axis.
            y_label: Label for Y-axis.
            z_label: Label for Z-axis.
            marker_color: Custom color for markers. Can be a single color (e.g., 'red', '#00FF00') 
                        or an array of colors matching the number of data points.
            marker_size: Size of the markers.
            marker_opacity: Opacity of the markers.
            interact_mode: Interaction mode ('hover' or 'click').
            run_server: Whether to run the Dash server.

        Returns:
            dash.Dash: A Dash application instance for the interactive plot.

        Raises:
            ValueError: If invalid options are selected or required methods haven't been run.
        """
        # Determine which data to use
        if which_data == 'original':
            vis_data = self.data
        elif which_data == 'reduced':
            if self.reduced_data is None:
                warnings.warn("'which_data' is set to 'reduced', but 'reduced_data' is not set. To show reduced data, either set the 'reduced_data' property of this object manually or generate it by running the 'reduce' method. Showing the 'original' data instead.", stacklevel=2)
                vis_data = self.data
            else:
                vis_data = self.reduced_data
        else:
            raise ValueError("which_data must be either 'original' or 'reduced'")


        # Handle n_components
        if n_components > 3:
            warnings.warn(f"n_components ({n_components}) > 3, only the first 3 components will be shown.", stacklevel=2)
            n_components = 3
        if n_components != vis_data.shape[-1]:
            warnings.warn(f"n_components ({n_components}) is different than the number of data features ({vis_data.shape[-1]}), only the first {min(n_components, vis_data.shape[-1])} components will be shown.", stacklevel=2)
            n_components = min(n_components, vis_data.shape[-1])

        app = interactive_scatterplot(
            x=vis_data[:, 0],
            y=vis_data[:, 1] if n_components > 1 else None,
            z=vis_data[:, 2] if n_components > 2 else None,
            true_labels=self.true_labels,
            cluster_labels=self.cluster_labels,
            interact_fn=interact_fn,
            x_label=x_label,
            y_label=y_label,
            z_label=z_label,
            marker_color=marker_color,
            marker_size=marker_size,
            marker_opacity=marker_opacity,
            interact_mode=interact_mode,
            run_server=run_server
        )

        return app


def analyze_and_show(
    data: np.ndarray,
    n_components: int = 2,
    true_labels: Optional[np.ndarray] = None, 
    reduction_method: ReductionMethodType = 'tsne',
    clustering_method: ClusteringMethodType = 'dbscan',
    scoring_method: ScoreMethodType = 'silhouette', 
    reduction_kwargs: Optional[dict] = None,
    clustering_kwargs: Optional[dict] = None, 
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
        reduction_kwargs: Additional parameters for reduction method (optional).
        clustering_kwargs: Additional parameters for clustering method (optional).
        viz_func: Custom visualization function (optional).
        verbose: Whether to print progress information.

    Returns:
        An InterDimAnalysis object with the analysis results.
    """
    analysis = InterDimAnalysis(data, true_labels, viz_func, verbose)
    analysis.reduce(method=reduction_method, n_components=n_components, **(reduction_kwargs or {}))
    analysis.cluster(method=clustering_method, **(clustering_kwargs or {}))
    analysis.score(method=scoring_method)
    analysis.show(n_components=n_components)
    
    return analysis