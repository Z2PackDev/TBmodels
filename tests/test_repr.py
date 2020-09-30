#!/usr/bin/env python

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Test the __repr__ of the Model class.
"""

import pytest
import numpy as np  # pylint: disable=unused-import

from parameters import T_VALUES

import tbmodels  # pylint: disable=unused-import
from tbmodels._sparse_matrix import csr  # pylint: disable=unused-import


@pytest.mark.parametrize("t", T_VALUES)
def test_repr(t, get_model, compare_equal):
    """Check that the repr() of a Model gives a consistent result."""
    compare_equal(repr(get_model(*t)))
