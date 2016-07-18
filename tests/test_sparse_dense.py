#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    07.07.2016 01:05:06 CEST
# File:    test_sparse_dense.py

import pytest
import tbmodels
import numpy as np

from parameters import T_VALUES, KPT

@pytest.mark.parametrize('t1', T_VALUES)
def test_simple(t1, get_model):
    model1 = get_model(*t1, sparse=True)
    model2 = get_model(*t1, sparse=False)

    for k in KPT:
        assert np.isclose(model1.hamilton(k), model2.hamilton(k)).all()

@pytest.mark.parametrize('t1', T_VALUES)
def test_change_to_dense(t1, get_model, models_close):
    model1 = get_model(*t1, sparse=True)
    model2 = get_model(*t1, sparse=False)
    model1.set_sparse(False)
    assert models_close(model1, model2)
    
@pytest.mark.parametrize('t1', T_VALUES)
def test_change_to_sparse(t1, get_model, models_close):
    model1 = get_model(*t1, sparse=True)
    model2 = get_model(*t1, sparse=False)
    model2.set_sparse(True)
    assert models_close(model1, model2)

@pytest.mark.parametrize('hr_file', ['./samples/hr_hamilton.dat', './samples/wannier90_hr.dat', './samples/wannier90_hr_v2.dat'])
def test_hr(hr_file):
    model1 = tbmodels.Model.from_hr_file(hr_file, occ=28, sparse=False)
    model2 = tbmodels.Model.from_hr_file(hr_file, occ=28, sparse=True)
    
    for k in KPT:
        assert np.isclose(model1.hamilton(k), model2.hamilton(k)).all()
