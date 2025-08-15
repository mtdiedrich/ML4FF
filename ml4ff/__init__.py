"""ML4FF - Machine Learning for Fantasy Football

A comprehensive machine learning toolkit for fantasy football analysis and predictions.
"""

__version__ = "0.1.0"
__author__ = "ML4FF Team"
__email__ = "info@ml4ff.com"

from .data_acquisition import DataAcquisitionPipeline

# Optional imports for pipelines
__all__ = ["DataAcquisitionPipeline"]

try:
    from .pipelines.rookie_projection import RookieProjectionPipeline
    __all__.append("RookieProjectionPipeline")
except ImportError:
    pass

try:
    from .pipelines.player_breakout import PlayerBreakoutPipeline
    __all__.append("PlayerBreakoutPipeline")
except ImportError:
    pass

try:
    from .pipelines.player_dropoff import PlayerDropoffPipeline
    __all__.append("PlayerDropoffPipeline")
except ImportError:
    pass