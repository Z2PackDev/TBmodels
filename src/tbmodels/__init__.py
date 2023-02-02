# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""A tool for creating / loading and manipulating tight-binding models."""

import importlib.metadata

__version__ = importlib.metadata.version(__name__.replace(".", "-"))

# import order is important due to circular imports
from . import helpers
from . import exceptions
from ._tb_model import Model

from . import kdotp
from . import io

__all__ = ("helpers", "exceptions", "Model", "kdotp", "io")
