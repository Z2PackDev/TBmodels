#!/usr/bin/env python
"""Tests for the eigenval method."""

import pytest
from numpy.testing import assert_allclose

from parameters import KPT, T_VALUES


@pytest.mark.parametrize("kpt", KPT)
@pytest.mark.parametrize("t_values", T_VALUES)
def test_simple_eigenval(get_model, kpt, t_values, compare_isclose):
    model = get_model(*t_values)
    compare_isclose(model.eigenval(kpt))


@pytest.mark.parametrize("t_values", T_VALUES)
def test_parallel_hamilton(get_model, t_values):
    model = get_model(*t_values)
    assert_allclose(model.eigenval(KPT), [model.eigenval(k) for k in KPT])
