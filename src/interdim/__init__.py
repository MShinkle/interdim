from importlib.metadata import version

__version__ = version(__name__)

from .pipeline import InterDimAnalysis
from . import reduce
from . import cluster
from . import vis
from .vis import InteractionPlot