#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    05.10.2015 16:54:52 CEST
# File:    arithmetics.py

import pytest
import numpy as np

from models import get_model

t_values = [(t1, t2) for t1 in [-0.1, 0.2, 0.3, 0.9] for t2 in [0.2, 0.4, 0.5]]

kpt = [(0.1, 0.2, 0.7), (-0.3, 0.5, 0.2), (0., 0., 0.), (0.1, -0.9, -0.7)]

@pytest.mark.parametrize('t1', t_values)
@pytest.mark.parametrize('t2', t_values)
@pytest.mark.parametrize('k', kpt)
def test_add(t1, t2, k, get_model, compare_equal):
    m1 = get_model(*t1)
    m2 = get_model(*t2)
    m3 = m1 + m2
    compare_equal(m3.hamilton(k))

@pytest.mark.parametrize('t1', t_values)
@pytest.mark.parametrize('t2', t_values)
@pytest.mark.parametrize('k', kpt)
def test_sub(t1, t2, k, get_model, compare_equal):
    m1 = get_model(*t1)
    m2 = get_model(*t2)
    m3 = m1 - m2
    compare_equal(m3.hamilton(k))
    
@pytest.mark.parametrize('t1', t_values)
@pytest.mark.parametrize('t2', t_values)
@pytest.mark.parametrize('k', kpt)
def test_sub_2(t1, t2, k, get_model, compare_equal):
    m1 = get_model(*t1)
    m2 = get_model(*t2)
    m3 = -m1 - m2
    compare_equal(m3.hamilton(k))
    
@pytest.mark.parametrize('t', t_values)
@pytest.mark.parametrize('c', np.linspace(-2, 2, 5))
@pytest.mark.parametrize('k', kpt)
def test_mul(t, c, k, get_model, compare_equal):
    m = get_model(*t)
    m *= c
    compare_equal(m.hamilton(k))
    
@pytest.mark.parametrize('t', t_values)
@pytest.mark.parametrize('c', np.linspace(-2, 2, 4))
@pytest.mark.parametrize('k', kpt)
def test_div(t, c, k, get_model, compare_equal):
    m = get_model(*t)
    m /= c
    compare_equal(m.hamilton(k))
    
@pytest.mark.parametrize('t', t_values)
@pytest.mark.parametrize('c', np.linspace(-2, 2, 4))
@pytest.mark.parametrize('k', kpt)
def test_div_consistency(t, c, k, get_model, compare_equal):
    m = get_model(*t)
    m2 = m / c
    m3 = m * (1. / c)
    assert np.isclose(m3.hamilton(k), m2.hamilton(k)).all()
