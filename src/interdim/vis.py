from typing import Literal, Optional, Callable, Union
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

def create_scatter_plot(x, y, z, marker_color, marker_size, marker_opacity, x_label, y_label, z_label):
    """Helper function to create scatter plot"""
    marker_settings = {
        'size': marker_size,
        'opacity': marker_opacity,
    }
    if marker_color is not None:
        marker_settings['color'] = marker_color

    if y is None and z is None:
        # 1D scatter plot
        scatter_fig = go.Figure(data=[
            go.Scatter(x=x, y=np.zeros_like(x), mode='markers', marker=marker_settings, showlegend=False)
        ])
        scatter_fig.update_yaxes(visible=False)
    elif z is None:
        # 2D scatter plot
        scatter_fig = go.Figure(data=[
            go.Scatter(x=x, y=y, mode='markers', marker=marker_settings, showlegend=False)
        ])
    else:
        # 3D scatter plot
        scatter_fig = go.Figure(data=[
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

    scatter_fig.update_xaxes(title_text=x_label)
    scatter_fig.update_yaxes(title_text=y_label)
    scatter_fig.update_layout(
        margin=dict(l=40, r=40, b=40, t=40),
        hovermode='closest'
    )

    return scatter_fig

def interactive_scatterplot(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    z: Optional[np.ndarray] = None,
    true_labels: Optional[np.ndarray] = None,
    cluster_labels: Optional[np.ndarray] = None,
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
    Create an interactive scatterplot using Plotly and Dash.

    Args:
        x: X-axis data.
        y: Y-axis data (optional for 2D and 3D plots).
        z: Z-axis data (optional for 3D plots).
        true_labels: True labels for coloring (optional).
        cluster_labels: Cluster labels for coloring (optional).
        interact_fn: Function to call on interaction events.
        x_label: Label for X-axis.
        y_label: Label for Y-axis.
        z_label: Label for Z-axis.
        marker_color: Custom color for markers (optional).
        marker_size: Size of the markers.
        marker_opacity: Opacity of the markers.
        interact_mode: Interaction mode ('hover' or 'click').
        run_server: Whether to run the Dash server.

    Returns:
        A Dash application instance.
    """
    app = dash.Dash(__name__)

    # Determine available color options
    color_options = ['Custom']
    if true_labels is not None:
        color_options.append('True Labels')
    if cluster_labels is not None:
        color_options.append('Clusters')

    # Create the initial scatter plot
    scatter_fig = create_scatter_plot(x, y, z, marker_color, marker_size, marker_opacity, x_label, y_label, z_label)

    # Create the color selector
    color_selector = dcc.RadioItems(
        id='color-selector',
        options=[{'label': opt, 'value': opt} for opt in color_options],
        value='Custom',
        inline=True
    )

    # Create the secondary plot for interactivity
    interact_fig = go.Figure()
    interact_fig.update_layout(
        margin=dict(l=40, r=40, b=40, t=40),
    )

    if interact_fn is None:
        interact_fig.add_annotation(
            text="Pass an interact_fn argument to update this plot.",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=12),
            x=0.5, y=0.5
        )
        interact_fig.update_xaxes(visible=False)
        interact_fig.update_yaxes(visible=False)

    app.layout = html.Div([
        html.H4("Marker Colors:"),
        color_selector,
        html.Div(
            style={
                'display': 'flex',
                'justify-content': 'space-between',
                'width': '100%',
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
    ])

    @app.callback(
        Output('scatter-plot', 'figure'),
        Input('color-selector', 'value')
    )
    def update_color(selected_option):
        if selected_option == 'True Labels':
            selected_color = true_labels
        elif selected_option == 'Clusters':
            selected_color = cluster_labels
        else:
            selected_color = marker_color
        
        return create_scatter_plot(x, y, z, selected_color, marker_size, marker_opacity, x_label, y_label, z_label)

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