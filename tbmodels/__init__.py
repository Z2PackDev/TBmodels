r"""
TBmodels is a tool for creating / loading and manipulating tight-binding models.
"""

from ._version import __version__

# import order is important due to circular imports
from . import helpers
from ._tb_model import Model

from . import io
