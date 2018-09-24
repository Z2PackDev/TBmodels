r"""
TBmodels is a tool for creating / loading and manipulating tight-binding models.
"""

__version__ = '1.3.0b1'

# import order is important due to circular imports
from . import helpers
from ._tb_model import Model

from . import io
