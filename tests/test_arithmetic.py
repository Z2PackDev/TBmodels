#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    05.10.2015 16:54:52 CEST
# File:    arithmetics.py

import pytest
import numpy as np

from models import get_model
from parameters import T_VALUES, KPT

@pytest.mark.parametrize('t1', T_VALUES)
@pytest.mark.parametrize('t2', T_VALUES)
@pytest.mark.parametrize('k', KPT)
def test_add(t1, t2, k, get_model, compare_equal):
    m1 = get_model(*t1)
    m2 = get_model(*t2)
    m3 = m1 + m2
    compare_equal(m3.hamilton(k))
    compare_equal(m3.eigenval(k), tag='eigenval')
    m4 = get_model(*t1, sparse=False)
    m5 = get_model(*t2, sparse=False)
    m6 = m4 + m5
    assert np.isclose(m3.hamilton(k), m6.hamilton(k)).all()

@pytest.mark.parametrize('t1', T_VALUES)
@pytest.mark.parametrize('t2', T_VALUES)
@pytest.mark.parametrize('k', KPT)
def test_sub(t1, t2, k, get_model, compare_equal):
    m1 = get_model(*t1)
    m2 = get_model(*t2)
    m3 = m1 - m2
    compare_equal(m3.hamilton(k))
    compare_equal(m3.eigenval(k), tag='eigenval')
    
@pytest.mark.parametrize('t1', T_VALUES)
@pytest.mark.parametrize('t2', T_VALUES)
@pytest.mark.parametrize('k', KPT)
def test_sub_2(t1, t2, k, get_model, compare_equal):
    m1 = get_model(*t1)
    m2 = get_model(*t2)
    m3 = -m1 - m2
    compare_equal(m3.hamilton(k))
    compare_equal(m3.eigenval(k), tag='eigenval')
    
@pytest.mark.parametrize('t', T_VALUES)
@pytest.mark.parametrize('c', np.linspace(-1, 1, 3))
@pytest.mark.parametrize('k', KPT)
def test_mul(t, c, k, get_model, compare_equal):
    m = get_model(*t)
    m *= c
    compare_equal(m.hamilton(k))
    compare_equal(m.eigenval(k), tag='eigenval')
    
@pytest.mark.parametrize('t', T_VALUES)
@pytest.mark.parametrize('c', np.linspace(-1, 0.5, 3)) # should be non-zero
@pytest.mark.parametrize('k', KPT)
def test_div(t, c, k, get_model, compare_equal):
    m = get_model(*t)
    m /= c
    compare_equal(m.hamilton(k))
    compare_equal(m.eigenval(k), tag='eigenval')
    
@pytest.mark.parametrize('t', T_VALUES)
@pytest.mark.parametrize('c', np.linspace(-1, 0.5, 3)) # should be non-zero
@pytest.mark.parametrize('k', KPT)
def test_div_consistency(t, c, k, get_model, compare_equal):
    m = get_model(*t)
    m2 = m / c
    m3 = m * (1. / c)
    assert np.isclose(m3.hamilton(k), m2.hamilton(k)).all()
    assert np.isclose(m3.eigenval(k), m2.eigenval(k)).all()
