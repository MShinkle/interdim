import numpy as np
from interdim.viz import interactive_scatterplot

def test_interactive_scatterplot_creation():
    x = np.random.rand(100)
    y = np.random.rand(100)
    app = interactive_scatterplot(x, y, run_server=False)
    assert app is not None