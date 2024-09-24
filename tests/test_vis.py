import numpy as np
import plotly.graph_objects as go
from interdim.vis import interactive_scatterplot

def test_interactive_scatterplot_creation():
    x = np.random.rand(100)
    y = np.random.rand(100)
    fig = interactive_scatterplot(x, y)
    assert isinstance(fig, go.Figure)