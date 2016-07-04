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

T1 = (0.1, 0.2)

def test_add_invalid_type(get_model, compare_equal):
    m = get_model(*T1)
    with pytest.raises(ValueError):
        m2 = m + 2

def test_add_invalid_occ(get_model, compare_equal):
    m1 = get_model(*T1)
    m2 = get_model(*T1, occ=2)
    with pytest.raises(ValueError):
        m3 = m1 + m2
