from typing import Any, Callable, Dict, List, Literal, Optional, Union

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Patch, dcc, html
from dash.dependencies import Input, Output, State


def create_scatter_plot(x, y, z, marker_color, marker_size, marker_opacity):
    """Helper function to create scatter plot"""
    marker_settings = {
        "size": marker_size,
        "opacity": marker_opacity,
    }
    if marker_color is not None:
        marker_settings["color"] = marker_color

    if y is None and z is None:
        # 1D scatter plot
        scatter_fig = go.Figure(
            data=[
                go.Scatter(
                    x=x,
                    y=np.zeros_like(x),
                    mode="markers",
                    marker=marker_settings,
                    showlegend=False,
                )
            ]
        )
        scatter_fig.update_yaxes(visible=False)
    elif z is None:
        # 2D scatter plot
        scatter_fig = go.Figure(
            data=[
                go.Scatter(
                    x=x, y=y, mode="markers", marker=marker_settings, showlegend=False
                )
            ]
        )
    else:
        # 3D scatter plot
        scatter_fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=marker_settings,
                    showlegend=False,
                )
            ]
        )
        scatter_fig.update_layout(
            scene=dict(
                xaxis_title=None,
                yaxis_title=None,
                zaxis_title=None,
                xaxis=dict(showspikes=False),
                yaxis=dict(showspikes=False),
                zaxis=dict(showspikes=False),
            )
        )

    scatter_fig.update_xaxes(title_text=None)
    scatter_fig.update_yaxes(title_text=None)
    scatter_fig.update_layout(margin=dict(l=40, r=40, b=40, t=40), hovermode="closest")

    return scatter_fig


def interactive_scatterplot(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    z: Optional[np.ndarray] = None,
    true_labels: Optional[np.ndarray] = None,
    cluster_labels: Optional[np.ndarray] = None,
    point_visualization: Optional[Union[Callable, str]] = None,
    marker_color: Optional[Union[str, np.ndarray]] = None,
    marker_size: int = 5,
    marker_opacity: float = 0.5,
    interact_mode: Literal["hover", "click"] = "hover",
    run_server: bool = True,
) -> dash.Dash:
    app = dash.Dash(__name__)
    color_options = ["Default" if marker_color is None else "Custom"]
    if true_labels is not None:
        color_options.append("True Labels")
    if cluster_labels is not None:
        color_options.append("Clusters")

    grey_text_style = {"color": "#808080"}  # Medium grey color

    scatter_fig = create_scatter_plot(
        x, y, z, marker_color, marker_size, marker_opacity
    )
    color_selector = dcc.RadioItems(
        id="color-selector",
        options=[{"label": opt, "value": opt} for opt in color_options],
        value="Default" if marker_color is None else "Custom",
        inline=True,
        style=grey_text_style,
        labelStyle=grey_text_style,
    )

    interact_fig = go.Figure()
    interact_fig.update_layout(margin=dict(l=40, r=40, b=40, t=40))

    if point_visualization is None:
        interact_fig.add_annotation(
            text="Pass a point_visualization argument to update this plot.",
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=12),
            x=0.5,
            y=0.5,
        )
        interact_fig.update_xaxes(visible=False)
        interact_fig.update_yaxes(visible=False)

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H4(
                        "Marker Colors",
                        style={**grey_text_style, "margin-bottom": "5px"},
                    ),
                    color_selector,
                ],
                style={"margin-bottom": "10px"},
            ),  # Adjust this value to control space before the graphs
            html.Div(
                style={
                    "display": "flex",
                    "justify-content": "space-between",
                    "width": "100%",
                },
                children=[
                    dcc.Graph(
                        id="scatter-plot", figure=scatter_fig, style={"flex": "0 0 48%"}
                    ),
                    dcc.Graph(
                        id="interact-plot",
                        figure=interact_fig,
                        style={"flex": "0 0 48%"},
                    ),
                ],
            ),
        ]
    )

    @app.callback(
        Output("scatter-plot", "figure"),
        Input("color-selector", "value"),
        State("scatter-plot", "figure"),
    )
    def update_color(selected_option, current_figure):
        patched_figure = Patch()

        if selected_option == "True Labels":
            selected_color = true_labels
        elif selected_option == "Clusters":
            selected_color = cluster_labels
        elif selected_option == "Default":
            selected_color = None
        else:  # "Custom"
            selected_color = marker_color

        patched_figure["data"][0]["marker"]["color"] = selected_color
        return patched_figure

    if point_visualization:

        @app.callback(
            Output("interact-plot", "figure"),
            [
                Input(
                    "scatter-plot",
                    "hoverData" if interact_mode == "hover" else "clickData",
                )
            ],
        )
        def update_interact_plot(selected_data):
            if selected_data:
                point_info_key = (
                    "pointIndex"
                    if "pointIndex" in selected_data["points"][0]
                    else "pointNumber"
                )
                index = selected_data["points"][0][point_info_key]
                fig = go.Figure()
                point_visualization(index, fig)
                return fig
            return interact_fig

    if run_server:
        app.run_server(debug=True)

    return app


class InteractionPlot:
    def __init__(
        self,
        data_source: Union[Callable, np.ndarray, List],
        plot_type: Literal[
            "bar", "box", "histogram", "line", "violin", "heatmap", "image", "surface"
        ],
        trace_kwargs: Optional[Dict] = None,
        layout_kwargs: Optional[Dict] = None,
    ):
        self.data_source = data_source
        self.plot_type = plot_type
        self.trace_kwargs = trace_kwargs or {}
        self.layout_kwargs = layout_kwargs or {}

    def __call__(self, index: int, fig: go.Figure) -> go.Figure:
        sample = self.get_sample(index)
        return self._plot(sample, fig)

    def get_sample(self, index: int) -> Any:
        if callable(self.data_source):
            return self.data_source(index)
        elif isinstance(self.data_source, (list, np.ndarray)):
            return self.data_source[index]
        else:
            raise ValueError("data_source must be callable or indexable")

    def _plot(self, sample: Any, fig: go.Figure) -> go.Figure:
        if self.plot_type.lower() == "bar":
            fig.add_trace(
                go.Bar(x=list(range(len(sample))), y=sample, **self.trace_kwargs)
            )
        elif self.plot_type.lower() == "box":
            fig.add_trace(go.Box(y=sample, **self.trace_kwargs))
        elif self.plot_type.lower() == "histogram":
            fig.add_trace(go.Histogram(x=sample, **self.trace_kwargs))
        elif self.plot_type.lower() == "line":
            fig.add_trace(
                go.Scatter(x=list(range(len(sample))), y=sample, **self.trace_kwargs)
            )
        elif self.plot_type.lower() == "violin":
            fig.add_trace(go.Violin(y=sample, **self.trace_kwargs))
        elif self.plot_type.lower() == "heatmap":
            fig.add_trace(go.Heatmap(z=sample, **self.trace_kwargs))
        elif self.plot_type.lower() == "image":
            fig.add_trace(go.Image(z=sample, **self.trace_kwargs))
        elif self.plot_type.lower() == "surface":
            fig.add_trace(go.Surface(z=sample, **self.trace_kwargs))
        else:
            raise ValueError(f"Plot type '{self.plot_type}' is not supported.")
        fig.update_layout(**self.layout_kwargs)
        return fig
