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

@pytest.mark.parametrize('hr_file', ['./samples/hr_hamilton.dat'])
def test_consistency(hr_file):
    model = tbmodels.Model.from_hr(hr_file, occ=28)
    lines_new = model.to_hr().split('\n')
    with open(hr_file, 'r') as f:
        lines_old = [line.rstrip(' \r\n') for line in f.readlines()]
    assert lines_new[1:] == lines_old[1:]
