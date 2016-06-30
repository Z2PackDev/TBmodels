#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    05.05.2015 13:59:00 CEST
# File:    hr_hamilton.py

import pytest

import tbmodels
import numpy as np

from models import get_model
from parameters import T_VALUES, KPT

@pytest.mark.parametrize('t', T_VALUES)
def test_hr_print(t, get_model, compare_equal):
    model = get_model(*t)
    compare_equal(model.to_hr().splitlines()[1:]) # timestamp in first line isn't equal
    

