#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    06.07.2016 02:48:48 CEST
# File:    test_simple_model.py

import pytest

from models import get_model
from parameters import T_VALUES, KPT

@pytest.mark.parametrize('t1', T_VALUES)
@pytest.mark.parametrize('t2', T_VALUES)
def test_add(t1, t2, get_model, compare_data, models_equal):
    m = get_model(*t1)
    compare_data(models_equal, m)
