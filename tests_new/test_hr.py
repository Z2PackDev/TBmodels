#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    05.05.2015 13:59:00 CEST
# File:    hr_hamilton.py

import pytest

import tbmodels
import numpy as np

kpt = [(0.1, 0.2, 0.7), (-0.3, 0.5, 0.2), (0., 0., 0.), (0.1, -0.9, -0.7)]

@pytest.mark.parametrize('hr_file', ['./samples/hr_hamilton.dat', './samples/wannier90_hr.dat'])
def test_hr(compare_data, hr_file):
    model = tbmodels.HrModel(hr_file, occ=28)
    H_list = np.array([model.hamilton(k) for k in kpt])

    compare_data(lambda x, y: np.isclose(x, y).all(), H_list)
        
def test_error():
    with pytest.raises(ValueError):
        tbmodels.HrModel('./samples/hr_hamilton.dat', occ=28, pos=[[1., 1., 1.]])
