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

@pytest.mark.parametrize('hr_file', ['./samples/hr_hamilton.dat', './samples/wannier90_hr.dat', './samples/wannier90_hr_v2.dat', './samples/silicon_hr.dat'])
def test_wannier_hr_only(compare_data, hr_file):
    model = tbmodels.Model.from_wannier_files(hr_file=hr_file, occ=28)
    H_list = np.array([model.hamilton(k) for k in kpt])

    compare_data(lambda x, y: np.isclose(x, y).all(), H_list)
    
@pytest.mark.parametrize('hr_file, wsvec_file', [
    ('./samples/silicon_hr.dat', './samples/silicon_wsvec.dat')
])
def test_wannier_hr_wsvec(compare_data, hr_file, wsvec_file):
    model = tbmodels.Model.from_wannier_files(hr_file=hr_file, wsvec_file=wsvec_file)
    H_list = np.array([model.hamilton(k) for k in kpt])

    compare_data(lambda x, y: np.isclose(x, y).all(), H_list)
        

@pytest.mark.parametrize('hr_file', ['./samples/hr_hamilton.dat', './samples/wannier90_hr.dat', './samples/wannier90_hr_v2.dat'])
def test_wannier_hr_equal(models_equal, hr_file):
    model1 = tbmodels.Model.from_hr_file(hr_file, occ=28)
    model2 = tbmodels.Model.from_wannier_files(hr_file=hr_file, occ=28)
    models_equal(model1, model2)

@pytest.mark.parametrize('hr_file', ['./samples/wannier90_inconsistent.dat', './samples/wannier90_inconsistent_v2.dat'])
def test_inconsistent(hr_file):
    with pytest.raises(ValueError):
        model = tbmodels.Model.from_wannier_files(hr_file=hr_file)


def test_emptylines():
    """test whether the input file with some random empty lines is correctly parsed"""
    model1 = tbmodels.Model.from_wannier_files(hr_file='./samples/wannier90_hr.dat')
    model2 = tbmodels.Model.from_wannier_files(hr_file='./samples/wannier90_hr_v2.dat')
    hop1 = model1.hop
    hop2 = model2.hop
    for k in hop1.keys() | hop2.keys():
        assert (np.array(hop1[k]) == np.array(hop2[k])).all()

def test_error():
    with pytest.raises(ValueError):
        tbmodels.Model.from_wannier_files(hr_file='./samples/hr_hamilton.dat', occ=28, pos=[[1., 1., 1.]])
