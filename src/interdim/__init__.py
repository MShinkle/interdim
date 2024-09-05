from importlib.metadata import version

__version__ = version(__name__)

from .pipeline import InterDimAnalysis, analyze_and_show
from . import reduce
from . import cluster
from . import viz