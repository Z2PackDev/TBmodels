#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""Tests for arithmetic operations on tight-binding models."""

import pytest
import numpy as np

from parameters import T_VALUES, KPT


@pytest.mark.parametrize("t1", T_VALUES)
@pytest.mark.parametrize("t2", T_VALUES)
def test_add(t1, t2, get_model, compare_isclose, sparse):
    """Basic test for adding two models."""
    model1 = get_model(*t1)
    model2 = get_model(*t2)
    sum_model = model1 + model2
    compare_isclose([sum_model.hamilton(k) for k in KPT])
    compare_isclose([sum_model.eigenval(k) for k in KPT], tag="eigenval")
    if sparse:
        model1_dense = get_model(*t1, sparse=False)
        model2_dense = get_model(*t2, sparse=False)
        sum_model_dense = model1_dense + model2_dense
        for k in KPT:
            assert np.isclose(sum_model.hamilton(k), sum_model_dense.hamilton(k)).all()


@pytest.mark.parametrize("t1", T_VALUES)
@pytest.mark.parametrize("t2", T_VALUES)
def test_sub(t1, t2, get_model, compare_isclose, sparse):
    """Basic test for subtracting two models."""
    model1 = get_model(*t1)
    model2 = get_model(*t2)
    sub_model = model1 - model2
    compare_isclose([sub_model.hamilton(k) for k in KPT])
    compare_isclose([sub_model.eigenval(k) for k in KPT], tag="eigenval")
    if sparse:
        model1_dense = get_model(*t1, sparse=False)
        model2_dense = get_model(*t2, sparse=False)
        sub_model_dense = model1_dense - model2_dense
        for k in KPT:
            assert np.isclose(sub_model.hamilton(k), sub_model_dense.hamilton(k)).all()


@pytest.mark.parametrize("t1", T_VALUES)
@pytest.mark.parametrize("t2", T_VALUES)
def test_sub_2(t1, t2, get_model, compare_isclose, sparse):
    """Test subtracting two modules, and unary negation."""
    model1 = get_model(*t1)
    model2 = get_model(*t2)
    res_model = -model1 - model2
    compare_isclose([res_model.hamilton(k) for k in KPT])
    compare_isclose([res_model.eigenval(k) for k in KPT], tag="eigenval")
    if sparse:
        model1_dense = get_model(*t1, sparse=False)
        model2_dense = get_model(*t2, sparse=False)
        res_model_dense = -model1_dense - model2_dense
        for k in KPT:
            assert np.isclose(res_model.hamilton(k), res_model_dense.hamilton(k)).all()


@pytest.mark.parametrize("t", T_VALUES)
@pytest.mark.parametrize("c", np.linspace(-1, 1, 3))
def test_mul(t, c, get_model, compare_isclose, sparse):  # pylint: disable=invalid-name
    """Test multiplying by a scalar."""
    model = get_model(*t)
    model *= c
    compare_isclose([model.hamilton(k) for k in KPT])
    compare_isclose([model.eigenval(k) for k in KPT], tag="eigenval")
    if sparse:
        model_dense = get_model(*t, sparse=False)
        model_dense *= c
        for k in KPT:
            assert np.isclose(model.hamilton(k), model_dense.hamilton(k)).all()


@pytest.mark.parametrize("t", T_VALUES)
@pytest.mark.parametrize("c", np.linspace(-1, 0.5, 3))  # should be non-zero
def test_div(t, c, get_model, compare_isclose, sparse):  # pylint: disable=invalid-name
    """Test dividing by a scalar."""
    model = get_model(*t)
    model /= c
    compare_isclose([model.hamilton(k) for k in KPT])
    compare_isclose([model.eigenval(k) for k in KPT], tag="eigenval")
    if sparse:
        model_dense = get_model(*t, sparse=False)
        model_dense /= c
        for k in KPT:
            assert np.isclose(model.hamilton(k), model_dense.hamilton(k)).all()


@pytest.mark.parametrize("t", T_VALUES)
@pytest.mark.parametrize("c", np.linspace(-1, 0.5, 3))  # should be non-zero
def test_div_consistency(t, c, get_model):  # pylint: disable=invalid-name
    """Test that dividing by a scalar is the same as multiplying by its inverse."""
    model = get_model(*t)
    model_div = model / c
    model_mul_inv = model * (1.0 / c)
    for k in KPT:
        assert np.isclose(model_div.hamilton(k), model_mul_inv.hamilton(k)).all()
        assert np.isclose(model_div.eigenval(k), model_mul_inv.eigenval(k)).all()
