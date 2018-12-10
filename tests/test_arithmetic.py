#!/usr/bin/env python
# -*- coding: utf-8 -*-

# © 2015-2018, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import pytest
import numpy as np

from parameters import T_VALUES, KPT


@pytest.mark.parametrize('t1', T_VALUES)
@pytest.mark.parametrize('t2', T_VALUES)
def test_add(t1, t2, get_model, compare_isclose):
    m1 = get_model(*t1)
    m2 = get_model(*t2)
    m3 = m1 + m2
    compare_isclose([m3.hamilton(k) for k in KPT])
    compare_isclose([m3.eigenval(k) for k in KPT], tag='eigenval')
    m4 = get_model(*t1, sparse=False)
    m5 = get_model(*t2, sparse=False)
    m6 = m4 + m5
    for k in KPT:
        assert np.isclose(m3.hamilton(k), m6.hamilton(k)).all()


@pytest.mark.parametrize('t1', T_VALUES)
@pytest.mark.parametrize('t2', T_VALUES)
def test_sub(t1, t2, get_model, compare_isclose):
    m1 = get_model(*t1)
    m2 = get_model(*t2)
    m3 = m1 - m2
    compare_isclose([m3.hamilton(k) for k in KPT])
    compare_isclose([m3.eigenval(k) for k in KPT], tag='eigenval')
    m4 = get_model(*t1, sparse=False)
    m5 = get_model(*t2, sparse=False)
    m6 = m4 - m5
    for k in KPT:
        assert np.isclose(m3.hamilton(k), m6.hamilton(k)).all()


@pytest.mark.parametrize('t1', T_VALUES)
@pytest.mark.parametrize('t2', T_VALUES)
def test_sub_2(t1, t2, get_model, compare_isclose):
    m1 = get_model(*t1)
    m2 = get_model(*t2)
    m3 = -m1 - m2
    compare_isclose([m3.hamilton(k) for k in KPT])
    compare_isclose([m3.eigenval(k) for k in KPT], tag='eigenval')
    m4 = get_model(*t1, sparse=False)
    m5 = get_model(*t2, sparse=False)
    m6 = -m4 - m5
    for k in KPT:
        assert np.isclose(m3.hamilton(k), m6.hamilton(k)).all()


@pytest.mark.parametrize('t', T_VALUES)
@pytest.mark.parametrize('c', np.linspace(-1, 1, 3))
def test_mul(t, c, get_model, compare_isclose):
    m1 = get_model(*t)
    m1 *= c
    compare_isclose([m1.hamilton(k) for k in KPT])
    compare_isclose([m1.eigenval(k) for k in KPT], tag='eigenval')
    m2 = get_model(*t, sparse=False)
    m2 *= c
    for k in KPT:
        assert np.isclose(m1.hamilton(k), m2.hamilton(k)).all()


@pytest.mark.parametrize('t', T_VALUES)
@pytest.mark.parametrize('c', np.linspace(-1, 0.5, 3))  # should be non-zero
def test_div(t, c, get_model, compare_isclose):
    m1 = get_model(*t)
    m1 /= c
    compare_isclose([m1.hamilton(k) for k in KPT])
    compare_isclose([m1.eigenval(k) for k in KPT], tag='eigenval')
    m2 = get_model(*t, sparse=False)
    m2 /= c
    for k in KPT:
        assert np.isclose(m1.hamilton(k), m2.hamilton(k)).all()


@pytest.mark.parametrize('t', T_VALUES)
@pytest.mark.parametrize('c', np.linspace(-1, 0.5, 3))  # should be non-zero
def test_div_consistency(t, c, get_model):
    m = get_model(*t)
    m2 = m / c
    m3 = m * (1. / c)
    for k in KPT:
        assert np.isclose(m3.hamilton(k), m2.hamilton(k)).all()
        assert np.isclose(m3.eigenval(k), m2.eigenval(k)).all()
