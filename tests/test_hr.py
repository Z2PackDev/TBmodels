#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""Tests for loading files from *_hr.dat format."""

import pytest
import numpy as np

import tbmodels

KPT = [(0.1, 0.2, 0.7), (-0.3, 0.5, 0.2), (0., 0., 0.), (0.1, -0.9, -0.7)]


@pytest.mark.parametrize('hr_name', ['hr_hamilton.dat', 'wannier90_hr.dat', 'wannier90_hr_v2.dat'])
def test_hr(compare_isclose, hr_name, sample):
    """
    Regression test for reading a model from *_hr.dat format.
    """
    hr_file = sample(hr_name)
    with pytest.deprecated_call():
        model = tbmodels.Model.from_hr_file(hr_file, occ=28)
    ham_list = np.array([model.hamilton(k) for k in KPT])
    compare_isclose(ham_list)


@pytest.mark.parametrize('hr_name', ['wannier90_inconsistent.dat', 'wannier90_inconsistent_v2.dat'])
def test_inconsistent(hr_name, sample):
    """
    Check that loading from incomplete / inconsistent *_hr.dat files
    raises an error.
    """
    with pytest.raises(ValueError):
        with pytest.deprecated_call():
            tbmodels.Model.from_hr_file(sample(hr_name))


def test_emptylines(sample):
    """test whether the input file with some random empty lines is correctly parsed"""
    with pytest.deprecated_call():
        model1 = tbmodels.Model.from_hr_file(sample('wannier90_hr.dat'))
    with pytest.deprecated_call():
        model2 = tbmodels.Model.from_hr_file(sample('wannier90_hr_v2.dat'))
    hop1 = model1.hop
    hop2 = model2.hop
    for k in hop1.keys() | hop2.keys():
        assert (np.array(hop1[k]) == np.array(hop2[k])).all()


def test_error(sample):
    """Check that an error is raised when an invalid number of positions is passed."""
    with pytest.raises(ValueError):
        with pytest.deprecated_call():
            tbmodels.Model.from_hr_file(sample('hr_hamilton.dat'), occ=28, pos=[[1., 1., 1.]])
