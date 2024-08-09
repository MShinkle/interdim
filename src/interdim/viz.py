import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

def interactive_scatterplot(x, y, z=None, interact_fn=None, x_label=None, y_label=None, z_label=None, marker_color='blue', marker_size=5, marker_opacity=.2, interact_mode='hover'):
    marker_settings = {
        'color': marker_color,  # Can be a single color or a list of colors
        'size': marker_size,    # Can be a single size or a list of sizes
        'opacity': marker_opacity,  # Can be a single opacity or a list of opacities
    }

    if z is None:
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
            xaxis=dict(showspikes=False),  # Hide spikes on x-axis
            yaxis=dict(showspikes=False),  # Hide spikes on y-axis
            zaxis=dict(showspikes=False),   # Hide spikes on z-axis
        ))

    scatter_fig.layout.hovermode = 'closest'
    scatter_fig.layout.margin = go.layout.Margin(l=20, r=20, b=20, t=20)
    scatter_fig.layout.height = 600
    scatter_fig.layout.width = 600

    # Secondary plot for interactivity
    interact_fig = go.FigureWidget()
    interact_fig.layout.margin = go.layout.Margin(l=20, r=20, b=20, t=20)
    interact_fig.layout.height = 600
    interact_fig.layout.width = 600

    if interact_fn is None:
        # Display a message in the interact figure if no interact_fn is provided
        interact_fig.add_annotation(
            text="Pass an interact_fn argument to update this plot.",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=12),
            x=0.5, y=0.5
        )
        # Hide x and y ticks
        interact_fig.update_xaxes(visible=False)
        interact_fig.update_yaxes(visible=False)
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
            'justify-content': 'space-between',  # Space between the figures
            'width': '800px',  # Fixed width of the container
            # 'margin': '0 auto'  # Center the container horizontally
        },
        children=[
            dcc.Graph(
                id='scatter-plot', 
                figure=scatter_fig,
                style={'flex': '0 0 48%'}  # Fixed size for each graph (can be adjusted)
            ),
            dcc.Graph(
                id='interact-plot', 
                figure=interact_fig,
                style={'flex': '0 0 48%'}  # Fixed size for each graph (can be adjusted)
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
                # Use 'pointIndex' for 2D and 'pointNumber' for 3D plots
                point_info_key = 'pointIndex' if 'pointIndex' in selected_data['points'][0] else 'pointNumber'
                index = selected_data['points'][0][point_info_key]
                interact_fn(index, interact_fig)
            return interact_fig

    app.run_server(debug=True)
