from typing import Any, Callable, Dict, Literal, Optional, Union, List

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Patch, dcc, html
from dash.dependencies import Input, Output, State
import socket


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def create_scatter_plot(x, y, z, marker_kwargs, scatter_kwargs=None):
    """Helper function to create scatter plot"""
    # Default scatter kwargs
    default_scatter_kwargs = {
        "mode": "markers",
        "marker": marker_kwargs,
        "showlegend": False,
    }

    # Update default_scatter_kwargs with user-provided scatter_kwargs
    if scatter_kwargs:
        default_scatter_kwargs.update(scatter_kwargs)

    if y is None and z is None:
        # 1D scatter plot
        default_scatter_kwargs.update({
            "x": x,
            "y": np.zeros_like(x),
        })
        scatter_fig = go.Figure(data=[go.Scatter(**default_scatter_kwargs)])
        scatter_fig.update_yaxes(visible=False)
    elif z is None:
        # 2D scatter plot
        default_scatter_kwargs.update({
            "x": x,
            "y": y,
        })
        scatter_fig = go.Figure(data=[go.Scatter(**default_scatter_kwargs)])
    else:
        # 3D scatter plot
        default_scatter_kwargs.update({
            "x": x,
            "y": y,
            "z": z,
        })
        scatter_fig = go.Figure(data=[go.Scatter3d(**default_scatter_kwargs)])
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
    marker_kwargs: Optional[Dict] = None,
    scatter_kwargs: Optional[Dict] = None,
    interact_mode: Literal["hover", "click"] = "hover",
    port: Optional[int] = None,
) -> dash.Dash:
    app = dash.Dash(__name__)
    if marker_kwargs is None:
        marker_kwargs = {}
    color_options = [
        "Default" if marker_kwargs is None or "color" not in marker_kwargs else "Custom"
    ]
    if true_labels is not None:
        color_options.append("True Labels")
    if cluster_labels is not None:
        color_options.append("Clusters")

    grey_text_style = {"color": "#808080"}  # Medium grey color

    scatter_fig = create_scatter_plot(x, y, z, marker_kwargs, scatter_kwargs)
    color_selector = dcc.RadioItems(
        id="color-selector",
        options=[{"label": opt, "value": opt} for opt in color_options],
        value="True Labels" if true_labels is not None else color_options[0],
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
            ),
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
    def update_marker_color(selected_color, current_figure):
        patched_figure = Patch()
        if selected_color == "True Labels" and true_labels is not None:
            patched_figure["data"][0]["marker"]["color"] = true_labels
        elif selected_color == "Clusters" and cluster_labels is not None:
            patched_figure["data"][0]["marker"]["color"] = cluster_labels
            patched_figure["data"][0]["marker"]["colorscale"] = marker_kwargs.get(
                "colorscale", "Viridis"
            )
        else:
            patched_figure["data"][0]["marker"]["color"] = marker_kwargs.get(
                "color", "blue"
            )
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

    if port is None:
        port = find_free_port()
    app.run_server(debug=True, port=port)

    return app


class InteractionPlot:
    def __init__(
        self,
        data_source,
        plot_type="bar",
        trace_kwargs=None,
        layout_kwargs=None,
        format_data=True,
    ):
        self.data_source = data_source
        self.plot_type = plot_type
        self.trace_kwargs = trace_kwargs or {}
        self.layout_kwargs = layout_kwargs or {}
        self.format_data = format_data

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

    def _format_image_data(self, sample: np.ndarray) -> np.ndarray:
        """Process the input array to fit the go.Image format (height x width x 3)."""
        sample = np.asarray(sample)  # Ensure input is a numpy array

        # Check if the data is float and in [0, 1] range
        if np.issubdtype(sample.dtype, np.floating) and sample.max() <= 1.0:
            sample = (sample * 255).astype(np.uint8)

        # Ensure 3D array
        if sample.ndim == 2:
            sample = sample[..., np.newaxis]

        # Ensure channels last
        if sample.shape[-1] not in (1, 3) and sample.shape[0] in (1, 3):
            sample = np.moveaxis(sample, 0, -1)

        # Ensure 3 channels
        if sample.shape[-1] == 1:
            sample = np.repeat(sample, 3, axis=-1)

        if sample.ndim != 3 or sample.shape[-1] != 3:
            raise ValueError(
                f"Unable to process image shape: {sample.shape}. "
                "Expected a 2D array or a 3D array with shape (height, width, channels). "
                "For 3D arrays, the last dimension should be 1 or 3 (grayscale or RGB), "
                "or the first dimension should be 1 or 3 (channel-first format). "
                f"Current shape: {sample.shape}"
            )

        return sample

    def _plot(self, sample: Any, fig: go.Figure) -> go.Figure:
        if self.plot_type.lower() == "text":
            default_trace_kwargs = {
                "x": [0],
                "y": [0],
                "mode": "text",
                "text": [sample],
                "textposition": "middle center",
                "textfont": dict(size=12),
            }
            default_trace_kwargs.update(self.trace_kwargs)
            fig.add_trace(go.Scatter(**default_trace_kwargs))
            fig.update_xaxes(visible=False, range=[-1, 1])
            fig.update_yaxes(visible=False, range=[-1, 1])
        elif self.plot_type.lower() == "image":
            if self.format_data:
                sample = self._format_image_data(sample)
            fig.add_trace(go.Image(z=sample, **self.trace_kwargs))
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
        elif self.plot_type.lower() == "bar":
            fig.add_trace(go.Bar(y=sample, **self.trace_kwargs))
            fig.update_layout(xaxis_tickangle=-45)
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
        elif self.plot_type.lower() == "surface":
            fig.add_trace(go.Surface(z=sample, **self.trace_kwargs))
        else:
            raise ValueError(f"Plot type '{self.plot_type}' is not supported.")

        fig.update_layout(**self.layout_kwargs)
        return fig
