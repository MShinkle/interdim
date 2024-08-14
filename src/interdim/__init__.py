from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

from .pipeline import InterDimAnalysis, analyze_and_visualize
from . import reduce
from . import cluster
from . import viz