import plotly.graph_objects as go
import ipywidgets as widgets

def interactive_scatterplot(x, y, on_click_fn, x_label=None, y_label=None, marker_color='blue', marker_size=10, marker_opacity=.2):
    # Prepare marker settings
    marker_settings = {
        'color': marker_color,  # Can be a single color or a list of colors
        'size': marker_size,    # Can be a single size or a list of sizes
        'opacity': marker_opacity,
    }

    # Create the primary scatter plot
    scatter_fig = go.FigureWidget(data=[
        go.Scatter(x=x, y=y, mode='markers', marker=marker_settings, showlegend=False)  # Remove legend for scatter points
    ])
    scatter_fig.layout.hovermode = 'closest'
    scatter_fig.layout.margin = go.layout.Margin(l=20, r=20, b=20, t=20)
    scatter_fig.layout.height = 500
    scatter_fig.layout.width = 500
    scatter_fig.update_xaxes(title_text=x_label)
    scatter_fig.update_yaxes(title_text=y_label)

    # Create an empty secondary plot
    interact_fig = go.FigureWidget()
    interact_fig.layout.margin = go.layout.Margin(l=20, r=20, b=20, t=20)
    interact_fig.layout.height = 500
    interact_fig.layout.width = 500
    interact_fig.layout.xaxis.visible = False
    interact_fig.layout.yaxis.visible = False

    # Function to handle click events on the scatter plot
    def on_point_click(trace, points, selector):
        if points.point_inds:
            index = points.point_inds[0]
            on_click_fn(index, interact_fig)

    scatter_fig.data[0].on_hover(on_point_click)

    # Create and return the layout containing both figures
    layout = widgets.HBox([scatter_fig, interact_fig], layout=widgets.Layout(align_self='stretch'))
    return layout