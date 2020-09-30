#!/usr/bin/env python
"""Tests for the Hamiltonian method."""

import pytest
from numpy.testing import assert_allclose

from parameters import KPT, T_VALUES


@pytest.mark.parametrize("kpt", KPT)
@pytest.mark.parametrize("t_values", T_VALUES)
@pytest.mark.parametrize("convention", [1, 2])
def test_simple_hamilton(get_model, kpt, t_values, compare_isclose, convention):
    """
    Regression test for the Hamiltonian of a simple model.
    """
    model = get_model(*t_values)
    compare_isclose(model.hamilton(kpt, convention=convention))


@pytest.mark.parametrize("t_values", T_VALUES)
@pytest.mark.parametrize("convention", [1, 2])
def test_parallel_hamilton(get_model, t_values, convention):
    """
    Test that passing multiple k-points to the Hamiltonian gives the
    same results as evaluating them individually.
    """
    model = get_model(*t_values)
    assert_allclose(
        model.hamilton(KPT, convention=convention),
        [model.hamilton(k, convention=convention) for k in KPT],
    )


@pytest.mark.parametrize("convention", ["a", "1", None])
def test_invalid_convention(get_model, convention):
    """
    Test that giving an invalid 'convention' raises an error.
    """
    model = get_model(t1=0, t2=0.1)
    with pytest.raises(ValueError):
        model.hamilton((0, 0, 0), convention=convention)
