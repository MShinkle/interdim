import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

def interactive_scatterplot(x, y, interact_fn, x_label=None, y_label=None, marker_color='blue', marker_size=10, marker_opacity=.2, interact_mode='hover'):
    marker_settings = {
        'color': marker_color, # Can be a single color or a list of colors
        'size': marker_size, # Can be a single size or a list of sizes
        'opacity': marker_opacity, # Can be a single opacity or a list of opacities
    }

    # Primary scatter plot
    scatter_fig = go.FigureWidget(data=[
        go.Scatter(x=x, y=y, mode='markers', marker=marker_settings, showlegend=False)
    ])
    scatter_fig.layout.hovermode = 'closest'
    scatter_fig.layout.margin = go.layout.Margin(l=20, r=20, b=20, t=20)
    scatter_fig.layout.height = 500
    scatter_fig.layout.width = 500
    scatter_fig.update_xaxes(title_text=x_label)
    scatter_fig.update_yaxes(title_text=y_label)

    # Secondary plot for interactivity
    interact_fig = go.FigureWidget()
    interact_fig.layout.margin = go.layout.Margin(l=20, r=20, b=20, t=20)
    interact_fig.layout.height = 500
    interact_fig.layout.width = 500
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

    app.layout = html.Div([
        html.Div(
            dcc.Graph(id='scatter-plot', figure=scatter_fig),
            style={'display': 'inline-block', 'width': '49%'}
        ),
        html.Div(
            dcc.Graph(id='interact-plot', figure=interact_fig),
            style={'display': 'inline-block', 'width': '49%'}
        )
    ])

    @app.callback(
        Output('interact-plot', 'figure'),
        [Input('scatter-plot', 'hoverData' if interact_mode == 'hover' else 'clickData')]
    )
    def update_interact_plot(selected_data):
        if selected_data:
            index = selected_data['points'][0]['pointIndex']
            interact_fn(index, interact_fig)
        return interact_fig

    app.run_server(debug=True)