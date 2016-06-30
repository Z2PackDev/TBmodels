#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    15.10.2014 10:22:43 CEST
# File:    repr.py

import pytest
import tbmodels

import numpy as np

from models import get_model
from tbmodels._ptools.sparse_matrix import csr


from models import get_model
from parameters import T_VALUES, KPT

@pytest.mark.parametrize('t', T_VALUES)
def test_repr_reload(t, get_model):
    m1 = get_model(*t)
    m2 = eval(repr(m1))
    for k in KPT:
        assert np.isclose(m1.hamilton(k), m2.hamilton(k)).all()
