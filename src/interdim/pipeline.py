import warnings
from typing import Callable, Literal, Optional, Union

import dash
import numpy as np

from .cluster import ClusteringMethodType, cluster_data
from .reduce import ReductionMethodType, reduce_dimensionality
from .score import ScoreMethodType, score_clustering
from .vis import InteractionPlot, interactive_scatterplot


class InterDimAnalysis:
    def __init__(
        self,
        data: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
        verbose: bool = True,
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
        self, method: ReductionMethodType = "tsne", n_components: int = 2, **kwargs
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
                raise ValueError(
                    f"If method is None, the number of features ({self.data.shape[1]}) must match n_components ({n_components})"
                )
            self.reduced_data = self.data
            if self.verbose:
                print("Skipping dimensionality reduction as method is None")
        else:
            if self.verbose:
                if len(kwargs) == 0:
                    print(
                        f"Performing dimensionality reduction via {method.upper()} with default arguments."
                    )
                else:
                    print(
                        f"Performing dimensionality reduction via {method.upper()} with the following custom arguments:"
                    )
                    print(f"\t{kwargs}")

            self.reduced_data = reduce_dimensionality(
                self.data, method=method, n_components=n_components, **kwargs
            )

        if self.verbose:
            print(f"Reduced data shape: {self.reduced_data.shape}")

        return self.reduced_data

    def cluster(
        self,
        method: ClusteringMethodType = "dbscan",
        which_data: str = "reduced",
        **kwargs,
    ) -> np.ndarray:
        """
        Perform clustering on the chosen dataset (original or reduced).

        Args:
            method: Clustering method to use.
            which_data: Which dataset to use for clustering ('original' or 'reduced').
            **kwargs: Additional arguments for the clustering method.

        Returns:
            The cluster labels.

        Raises:
            ValueError: If the selected dataset hasn't been prepared.
        """
        # Determine which data to use for clustering
        if which_data == "original":
            clustering_data = self.data
        elif which_data == "reduced":
            if self.reduced_data is None:
                raise ValueError(
                    "'which_data' is set to 'reduced', but 'reduced_data' is not set. Please run the 'reduce' method first or use 'original' data."
                )
            clustering_data = self.reduced_data
        else:
            raise ValueError("which_data must be either 'original' or 'reduced'")

        if method is None:
            self.cluster_labels = None
            if self.verbose:
                print("Skipping clustering as method is None")
        else:
            if self.verbose:
                if len(kwargs) == 0:
                    print(
                        f"Performing clustering via {method.upper()} with default arguments."
                    )
                else:
                    print(
                        f"Performing clustering via {method.upper()} with the following custom arguments:"
                    )
                    print(f"\t{kwargs}")

            self.cluster_labels = cluster_data(clustering_data, method=method, **kwargs)

            if self.verbose:
                n_clusters = len(np.unique(self.cluster_labels))
                print(f"Clustering complete. Number of clusters: {n_clusters}")

        return self.cluster_labels

    def score(
        self,
        method: ScoreMethodType = "silhouette",
        true_labels: Optional[np.ndarray] = None,
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

        self.cluster_score = score_clustering(
            self.reduced_data, self.cluster_labels, labels_to_use, method
        )

        if self.verbose:
            print(f"Clustering Evaluation Result ({method}):")
            print(f"Score: {self.cluster_score}")

        return self.cluster_score

    def show(
        self,
        n_components: Optional[int] = 3,
        which_data: str = "reduced",
        point_visualization: Optional[
            Union[
                Callable,
                Literal[
                    "bar",
                    "box",
                    "histogram",
                    "line",
                    "violin",
                    "heatmap",
                    "image",
                    "surface",
                ],
            ]
        ] = None,
        marker_color: Optional[Union[str, np.ndarray]] = None,
        marker_size: int = 5,
        marker_opacity: float = 0.5,
        interact_mode: Literal["hover", "click"] = "hover",
        run_server: bool = True,
    ) -> dash.Dash:
        """
        Generate an interactive visualization of the data.

        Args:
            n_components: Number of components to show (1, 2, or 3).
            which_data: Which dataset to show ('original' or 'reduced').
            point_visualization: Either a function or a string specifying the plot type for interaction events.
            marker_color: Custom color for markers, can be a single color or an array of colors.
            marker_size: Size of the markers.
            marker_opacity: Opacity of the markers.
            interact_mode: Interaction mode ('hover' or 'click').
            run_server: Whether to run the Dash server.

        Returns:
            dash.Dash: A Dash application instance for the interactive plot.

        Raises:
            ValueError: If invalid options are selected or required methods haven't been run.
        """
        if which_data == "original":
            vis_data = self.data
        elif which_data == "reduced":
            if self.reduced_data is None:
                warnings.warn(
                    "'which_data' is set to 'reduced', but 'reduced_data' is not set. To show reduced data, either set the 'reduced_data' property of this object manually or generate it by running the 'reduce' method. Showing the 'original' data instead.",
                    stacklevel=2,
                )
                vis_data = self.data
            else:
                vis_data = self.reduced_data
        else:
            raise ValueError("which_data must be either 'original' or 'reduced'")

        if n_components > 3:
            warnings.warn(
                f"n_components ({n_components}) > 3, only the first 3 components will be shown.",
                stacklevel=2,
            )
            n_components = 3

        if n_components != vis_data.shape[-1]:
            warnings.warn(
                f"n_components ({n_components}) is different than the number of data features ({vis_data.shape[-1]}), only the first {min(n_components, vis_data.shape[-1])} components will be shown.",
                stacklevel=2,
            )
            n_components = min(n_components, vis_data.shape[-1])

        if isinstance(point_visualization, str):
            point_visualization = InteractionPlot(
                data_source=self.data, plot_type=point_visualization
            )

        app = interactive_scatterplot(
            x=vis_data[:, 0],
            y=vis_data[:, 1] if n_components > 1 else None,
            z=vis_data[:, 2] if n_components > 2 else None,
            true_labels=self.true_labels,
            cluster_labels=self.cluster_labels,
            point_visualization=point_visualization,
            marker_color=marker_color,
            marker_size=marker_size,
            marker_opacity=marker_opacity,
            interact_mode=interact_mode,
            run_server=run_server,
        )

        return app
