#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    05.05.2015 13:59:00 CEST
# File:    hr_hamilton.py

import tempfile
from os.path import join

import pytest

import tbmodels
import numpy as np

from parameters import T_VALUES, KPT, SAMPLES_DIR

@pytest.mark.parametrize('t', T_VALUES)
def test_hr_print(t, get_model, compare_equal):
    model = get_model(*t)
    compare_equal(model.to_hr().splitlines()[1:]) # timestamp in first line isn't equal

@pytest.mark.parametrize('hr_name', ['hr_hamilton.dat'])
def test_consistency(hr_name):
    hr_file = join(SAMPLES_DIR, hr_name)
    model = tbmodels.Model.from_hr_file(hr_file, occ=28, sparse=True)
    lines_new = model.to_hr().split('\n')
    with open(hr_file, 'r') as f:
        lines_old = [line.rstrip(' \r\n') for line in f.readlines()]
    assert lines_new[1:] == lines_old[1:]
    
@pytest.mark.parametrize('hr_name', ['hr_hamilton.dat'])
def test_consistency_file(hr_name, models_equal, sparse):
    hr_file = join(SAMPLES_DIR, hr_name)
    model1 = tbmodels.Model.from_hr_file(hr_file, sparse=sparse)
    with tempfile.NamedTemporaryFile() as tmpf:
        model1.to_hr_file(tmpf.name)
        model2 = tbmodels.Model.from_hr_file(tmpf.name, sparse=sparse)
    models_equal(model1, model2)
    
@pytest.mark.parametrize('hr_name', ['hr_hamilton.dat'])
def test_consistency_str(hr_name, models_equal, sparse):
    hr_file = join(SAMPLES_DIR, hr_name)
    model1 = tbmodels.Model.from_hr_file(hr_file, sparse=sparse)
    model2 = tbmodels.Model.from_hr(model1.to_hr(), sparse=sparse)
    models_equal(model1, model2)

@pytest.mark.parametrize('hr_name', ['hr_hamilton.dat', 'hr_hamilton_full.dat'])
def test_consistency_no_hcutoff(hr_name):
    hr_file = join(SAMPLES_DIR, hr_name)
    model = tbmodels.Model.from_hr_file(hr_file, occ=28, h_cutoff=-1, sparse=True)
    lines_new = model.to_hr().split('\n')
    with open(hr_file, 'r') as f:
        lines_old = [line.rstrip(' \r\n') for line in f.readlines()]
    assert lines_new[1:] == lines_old[1:]

def test_invalid_empty():
    model = tbmodels.Model(size=2, dim=3)
    with pytest.raises(ValueError):
        model.to_hr()
