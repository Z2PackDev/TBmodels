#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Test conversion of models into kwant format.
"""

import warnings
import itertools

import pytest
import numpy as np
import scipy.linalg as la

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import kwant
from kwant import wraparound

import tbmodels

from parameters import T_VALUES, KPT


def to_kwant_params(kval):
    """Helper function to turn a k-point value into a kwant params dict."""
    return {key: 2 * np.pi * val for key, val in zip(["k_x", "k_y", "k_z"], kval)}


@pytest.mark.parametrize("t", T_VALUES)
def test_simple(t, get_model):
    """
    Check converting a simple model to kwant. The models are checked for
    equivalence by wrapping around the kwant model and comparing the
    resulting Hamiltonians.
    """
    model = get_model(*t)

    latt = model.to_kwant_lattice()
    sym = kwant.TranslationalSymmetry(  # pylint: disable=no-member
        latt.vec((1, 0, 0)), latt.vec((0, 1, 0)), latt.vec((0, 0, 1))
    )
    sys = kwant.Builder(sym)  # pylint: disable=no-member
    sys[latt.shape(lambda p: True, (0, 0, 0))] = 0
    model.add_hoppings_kwant(sys)
    sys = wraparound.wraparound(sys).finalized()

    # the Hamiltonian doesn't match because the sites might be re-ordered -> test eigenval instead
    for k in KPT:
        np.testing.assert_allclose(
            model.eigenval(k),
            la.eigvalsh(sys.hamiltonian_submatrix(params=to_kwant_params(k))),
            atol=1e-8,
        )


@pytest.mark.parametrize(
    "hr_name", ["hr_hamilton.dat", "wannier90_hr.dat", "wannier90_hr_v2.dat"]
)
def test_realistic(hr_name, sample):
    """
    Check converting a realistic model to kwant. The models are checked for
    equivalence by wrapping around the kwant model and comparing the
    resulting Hamiltonians.
    """
    hr_file = sample(hr_name)
    model = tbmodels.Model.from_wannier_files(hr_file=hr_file, occ=28)

    latt = model.to_kwant_lattice()
    sym = kwant.TranslationalSymmetry(  # pylint: disable=no-member
        latt.vec((1, 0, 0)), latt.vec((0, 1, 0)), latt.vec((0, 0, 1))
    )
    sys = kwant.Builder(sym)  # pylint: disable=no-member
    sys[latt.shape(lambda p: True, (0, 0, 0))] = 0
    model.add_hoppings_kwant(sys)
    sys = wraparound.wraparound(sys).finalized()

    # don't split into separate tests because it takes too long
    # since there is only one 'site' we can also test the Hamiltonian
    for k in KPT:
        np.testing.assert_allclose(
            model.eigenval(k),
            la.eigvalsh(sys.hamiltonian_submatrix(params=to_kwant_params(k))),
            atol=1e-8,
        )
        np.testing.assert_allclose(
            model.hamilton(k),
            sys.hamiltonian_submatrix(params=to_kwant_params(k)),
            atol=1e-8,
        )


def test_unequal_orbital_number():
    """
    Check converting a simple model to kwant, where the two positions
    don't have an equal number of orbitals.
    """
    model = tbmodels.Model(
        pos=[[0.0, 0.0], [0.5, 0.5], [0.5, 0.5]], on_site=[1, 0.7, -1.2]
    )
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
    sym = kwant.TranslationalSymmetry(  # pylint: disable=no-member
        latt.vec((1, 0)), latt.vec((0, 1))
    )
    sys = kwant.Builder(sym)  # pylint: disable=no-member
    sys[latt.shape(lambda p: True, (0, 0))] = 0
    model.add_hoppings_kwant(sys)
    sys = wraparound.wraparound(sys).finalized()

    for k in KPT:
        k = k[:2]
        np.testing.assert_allclose(
            model.eigenval(k),
            la.eigvalsh(sys.hamiltonian_submatrix(params=to_kwant_params(k))),
            atol=1e-8,
        )
