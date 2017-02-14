#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    05.05.2015 13:59:00 CEST
# File:    hr_hamilton.py

import pytest

import kwant
import tbmodels
import wraparound
import numpy as np

from parameters import T_VALUES, KPT

#~ kpt = [(0.1, 0.2, 0.7), (-0.3, 0.5, 0.2), (0., 0., 0.), (0.1, -0.9, -0.7)]
#~ kpt_kwant = [tuple(2 * np.pi * np.array(k)) for k in kpt]


@pytest.mark.parametrize('t', T_VALUES)
def test_simple(t, get_model):
    model = get_model(*t)
    
    latt = model.to_kwant_lattice()
    sym = kwant.TranslationalSymmetry(
        latt.vec((1, 0, 0)),
        latt.vec((0, 1, 0)),
        latt.vec((0, 0, 1))
    )
    sys = kwant.Builder(sym)
    sys[latt.shape(lambda p: True, (0, 0, 0))] = 0
    model.add_hoppings_kwant(sys)
    sys = wraparound.wraparound(sys).finalized()
    
    for k in KPT:
        print(model.hamilton(k).shape)
        k_kwant = tuple(np.array(k) * 2 * np.pi)
        np.testing.assert_allclose(model.hamilton(k), sys.hamiltonian_submatrix(k_kwant), atol=1e-8)

@pytest.mark.parametrize('hr_file', ['./samples/hr_hamilton.dat', './samples/wannier90_hr.dat', './samples/wannier90_hr_v2.dat'])
def test_realistic(compare_data, hr_file):
    model = tbmodels.Model.from_hr_file(hr_file, occ=28)
    
    latt = model.to_kwant_lattice()
    sym = kwant.TranslationalSymmetry(
        latt.vec((1, 0, 0)),
        latt.vec((0, 1, 0)),
        latt.vec((0, 0, 1))
    )
    sys = kwant.Builder(sym)
    sys[latt.shape(lambda p: True, (0, 0, 0))] = 0
    model.add_hoppings_kwant(sys)
    sys = wraparound.wraparound(sys).finalized()
    
    # don't split into separate tests because it takes too long
    for k in KPT:
        k_kwant = tuple(np.array(k) * 2 * np.pi)
        np.testing.assert_allclose(model.hamilton(k), sys.hamiltonian_submatrix(k_kwant), atol=1e-8)
