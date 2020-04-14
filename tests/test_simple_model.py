#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""Tests for creating a simple tight-binding model."""

import pytest

from parameters import T_VALUES, KPT


@pytest.mark.parametrize("t1", T_VALUES)
@pytest.mark.parametrize("k", KPT)
def test_simple(t1, get_model, k, compare_data, models_equal, compare_isclose):
    """Regression test for a simple manually created tight-binding model."""
    model = get_model(*t1)

    compare_isclose(model.hamilton(k), tag="hamilton")
    compare_isclose(model.eigenval(k), tag="eigenval")
    compare_data(models_equal, model)


def test_invalid_dim(get_model):
    """
    Check that an error is raised when the reciprocal lattice vector
    does not match the dimension.
    """
    model = get_model(0.1, 0.2)
    model.add_hop(1j, 0, 1, (0, 1, 2))
    with pytest.raises(ValueError):
        model.add_hop(1j, 0, 1, (0, 1))
