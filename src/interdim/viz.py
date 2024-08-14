from typing import Literal, Optional, Callable
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

def interactive_scatterplot(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    z: Optional[np.ndarray] = None,
    interact_fn: Optional[Callable] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    z_label: Optional[str] = None,
    marker_color: str = 'blue',
    marker_size: int = 5,
    marker_opacity: float = 0.2,
    interact_mode: Literal["hover", "click"] = 'hover',
    run_server: bool = True
) -> dash.Dash:
    """
    Create an interactive scatterplot using Plotly and Dash.

    Args:
        x: X-axis data.
        y: Y-axis data (optional for 2D and 3D plots).
        z: Z-axis data (optional for 3D plots).
        interact_fn: Function to call on interaction events.
        x_label: Label for X-axis.
        y_label: Label for Y-axis.
        z_label: Label for Z-axis.
        marker_color: Color of the markers.
        marker_size: Size of the markers.
        marker_opacity: Opacity of the markers.
        interact_mode: Interaction mode ('hover' or 'click').
        run_server: Whether to run the Dash server.

    Returns:
        A Dash application instance.
    """
    marker_settings = {
        'color': marker_color,
        'size': marker_size,
        'opacity': marker_opacity,
    }

    if y is None and z is None:
        # 1D scatter plot
        scatter_fig = go.FigureWidget(data=[
            go.Scatter(x=x, y=np.zeros_like(x), mode='markers', marker=marker_settings, showlegend=False)
        ])
        scatter_fig.update_xaxes(title_text=x_label)
        scatter_fig.update_yaxes(title_text=y_label, visible=False)
    elif z is None:
        # 2D scatter plot
        scatter_fig = go.FigureWidget(data=[
            go.Scatter(x=x, y=y, mode='markers', marker=marker_settings, showlegend=False)
        ])
        scatter_fig.update_xaxes(title_text=x_label)
        scatter_fig.update_yaxes(title_text=y_label)
    else:
        # 3D scatter plot
        scatter_fig = go.FigureWidget(data=[
            go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=marker_settings, showlegend=False)
        ])
        scatter_fig.update_layout(scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
            xaxis=dict(showspikes=False),
            yaxis=dict(showspikes=False),
            zaxis=dict(showspikes=False),
        ))

    scatter_fig.layout.hovermode = 'closest'
    scatter_fig.layout.margin = go.layout.Margin(l=40, r=40, b=40, t=40)
    scatter_fig.layout.height = 600
    scatter_fig.layout.width = 600

    # Secondary plot for interactivity
    interact_fig = go.FigureWidget()
    interact_fig.layout.margin = go.layout.Margin(l=40, r=40, b=40, t=40)
    interact_fig.layout.height = 600
    interact_fig.layout.width = 600

    if interact_fn is None:
        interact_fig.add_annotation(
            text="Pass an interact_fn argument to update this plot.",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=12),
            x=0.5, y=0.5
        )
    else:
        interact_fig.layout.xaxis.visible = False
        interact_fig.layout.yaxis.visible = False

        def on_interact(trace, points, selector):
            if points.point_inds:
                index = points.point_inds[0]
                interact_fn(index, interact_fig)

        if interact_mode == 'hover':
            scatter_fig.data[0].on_hover(on_interact)
        elif interact_mode == 'click':
            scatter_fig.data[0].on_click(on_interact)
        else:
            raise ValueError(f"Invalid interact_mode: {interact_mode}. Expected 'hover' or 'click'.")

    app = dash.Dash(__name__)

    app.layout = html.Div(
        style={
            'display': 'flex',
            'justify-content': 'space-between',
            'width': '800px',
        },
        children=[
            dcc.Graph(
                id='scatter-plot', 
                figure=scatter_fig,
                style={'flex': '0 0 48%'}
            ),
            dcc.Graph(
                id='interact-plot', 
                figure=interact_fig,
                style={'flex': '0 0 48%'}
            )
        ]
    )

    if interact_fn:
        @app.callback(
            Output('interact-plot', 'figure'),
            [Input('scatter-plot', 'hoverData' if interact_mode == 'hover' else 'clickData')]
        )
        def update_interact_plot(selected_data):
            if selected_data:
                point_info_key = 'pointIndex' if 'pointIndex' in selected_data['points'][0] else 'pointNumber'
                index = selected_data['points'][0][point_info_key]
                interact_fn(index, interact_fig)
            return interact_fig

    if run_server:
        app.run_server(debug=True)
        
    return app