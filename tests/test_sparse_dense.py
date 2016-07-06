#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    07.07.2016 01:05:06 CEST
# File:    test_sparse_dense.py

import pytest
import tbmodels
import numpy as np

from models import get_model
from parameters import T_VALUES, KPT

@pytest.mark.parametrize('t1', T_VALUES)
@pytest.mark.parametrize('k', KPT)
def test_simple(t1, get_model, k):
    m1 = get_model(*t1, sparse=True)
    m2 = get_model(*t1, sparse=False)
    
    assert np.isclose(m1.hamilton(k), m2.hamilton(k)).all()
    
    #~ compare_equal(m.hamilton(k), tag='hamilton')
    #~ compare_equal(m.eigenval(k), tag='eigenval')
