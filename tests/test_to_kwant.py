#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import pytest
import itertools

import kwant
import tbmodels
import wraparound
import numpy as np
import scipy.linalg as la

from parameters import T_VALUES, KPT


@pytest.mark.parametrize('t', T_VALUES)
def test_simple(t, get_model):
    model = get_model(*t)

    latt = model.to_kwant_lattice()
    sym = kwant.TranslationalSymmetry(latt.vec((1, 0, 0)), latt.vec((0, 1, 0)), latt.vec((0, 0, 1)))
    sys = kwant.Builder(sym)
    sys[latt.shape(lambda p: True, (0, 0, 0))] = 0
    model.add_hoppings_kwant(sys)
    sys = wraparound.wraparound(sys).finalized()

    # the Hamiltonian doesn't match because the sites might be re-ordered -> test eigenval instead
    for k in KPT:
        k_kwant = tuple(np.array(k) * 2 * np.pi)
        np.testing.assert_allclose(model.eigenval(k), la.eigvalsh(sys.hamiltonian_submatrix(k_kwant)), atol=1e-8)


@pytest.mark.parametrize('hr_name', ['hr_hamilton.dat', 'wannier90_hr.dat', 'wannier90_hr_v2.dat'])
def test_realistic(hr_name, sample):
    hr_file = sample(hr_name)
    model = tbmodels.Model.from_hr_file(hr_file, occ=28)

    latt = model.to_kwant_lattice()
    sym = kwant.TranslationalSymmetry(latt.vec((1, 0, 0)), latt.vec((0, 1, 0)), latt.vec((0, 0, 1)))
    sys = kwant.Builder(sym)
    sys[latt.shape(lambda p: True, (0, 0, 0))] = 0
    model.add_hoppings_kwant(sys)
    sys = wraparound.wraparound(sys).finalized()

    # don't split into separate tests because it takes too long
    # since there is only one 'site' we can also test the Hamiltonian
    for k in KPT:
        k_kwant = tuple(np.array(k) * 2 * np.pi)
        np.testing.assert_allclose(model.eigenval(k), la.eigvalsh(sys.hamiltonian_submatrix(k_kwant)), atol=1e-8)
        np.testing.assert_allclose(model.hamilton(k), sys.hamiltonian_submatrix(k_kwant), atol=1e-8)


def test_unequal_orbital_number():
    model = tbmodels.Model(pos=[[0., 0.], [0.5, 0.5], [0.5, 0.5]], on_site=[1, 0.7, -1.2])
    t1 = 0.1
    t2 = 0.15
    t3 = 0.4
    for phase, R in zip([1, -1j, 1j, -1], itertools.product([0, -1], [0, -1])):
        model.add_hop(t1 * phase, 0, 1, R)
        model.add_hop(t3 * phase, 1, 2, R)

    for R in ((r[0], r[1]) for r in itertools.permutations([0, 1])):
        model.add_hop(t2, 0, 0, R)
        model.add_hop(-t2, 1, 1, R)
        model.add_hop(-t2, 2, 2, R)

    latt = model.to_kwant_lattice()
    sym = kwant.TranslationalSymmetry(latt.vec((1, 0)), latt.vec((0, 1)))
    sys = kwant.Builder(sym)
    sys[latt.shape(lambda p: True, (0, 0))] = 0
    model.add_hoppings_kwant(sys)
    sys = wraparound.wraparound(sys).finalized()

    for k in KPT:
        k = k[:2]
        k_kwant = tuple(np.array(k) * 2 * np.pi)
        np.testing.assert_allclose(model.eigenval(k), la.eigvalsh(sys.hamiltonian_submatrix(k_kwant)), atol=1e-8)
