#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for constructing supercell models."""

import itertools

import numpy as np
from numpy.testing import assert_allclose
import pytest

from parameters import KPT, T_VALUES

import tbmodels


def get_equivalent_k(k, supercell_size):
    return itertools.product(
        *[
            (np.linspace(0, 1, s, endpoint=False) + ki / s)
            for ki, s in zip(k, supercell_size)
        ]
    )


@pytest.mark.parametrize("t_values", T_VALUES)
@pytest.mark.parametrize("supercell_size", [(1, 1, 1), (2, 1, 1), (2, 3, 2)])
def test_supercell_simple(get_model, t_values, supercell_size, sparse):
    """
    Test that the eigenvalues from a supercell model match the folded
    eigenvalues of the base model, for a simple model.
    """
    model = get_model(*t_values, sparse=sparse)
    supercell_model = model.supercell(size=supercell_size)
    for k in KPT:
        ev_supercell = supercell_model.eigenval(k)
        equivalent_k = get_equivalent_k(k, supercell_size)
        ev_folded = np.sort(
            np.array([model.eigenval(kval) for kval in equivalent_k]).flatten()
        )
        assert ev_supercell.shape == ev_folded.shape
        assert_allclose(ev_supercell, ev_folded, atol=1e-7)


@pytest.mark.parametrize("t_values", T_VALUES)
@pytest.mark.parametrize("supercell_size", [(5, 4), (1, 1), (2, 3)])
def test_supercell_simple_2d(get_model, t_values, supercell_size):
    """
    Test that the eigenvalues from a supercell model match the folded
    eigenvalues of the base model, for a simple model.
    """
    model = get_model(*t_values, dim=2)
    supercell_model = model.supercell(size=supercell_size)
    for k in [(-0.12341, 0.92435), (0, 0), (0.65432, -0.1561)]:
        ev_supercell = supercell_model.eigenval(k)
        equivalent_k = get_equivalent_k(k, supercell_size)
        ev_folded = np.sort(
            np.array([model.eigenval(kval) for kval in equivalent_k]).flatten()
        )
        assert ev_supercell.shape == ev_folded.shape
        assert_allclose(ev_supercell, ev_folded, atol=1e-7)


@pytest.mark.parametrize("t_values", T_VALUES)
@pytest.mark.parametrize("supercell_size", [(5, 4, 2, 2), (1, 1, 1, 1), (2, 2, 3, 2)])
def test_supercell_simple_4d(get_model, t_values, supercell_size):
    """
    Test that the eigenvalues from a supercell model match the folded
    eigenvalues of the base model, for a simple model.
    """
    model = get_model(*t_values, dim=4)
    supercell_model = model.supercell(size=supercell_size)
    for k in [
        (-0.12341, 0.92435, 0.32, 0.1212),
        (0, 0, 0, 0),
        (0.65432, -0.1561, 0.2352346, -0.92345),
    ]:
        ev_supercell = supercell_model.eigenval(k)
        equivalent_k = get_equivalent_k(k, supercell_size)
        ev_folded = np.sort(
            np.array([model.eigenval(kval) for kval in equivalent_k]).flatten()
        )
        assert ev_supercell.shape == ev_folded.shape
        assert_allclose(ev_supercell, ev_folded, atol=1e-7)


@pytest.mark.parametrize("supercell_size", [(1, 1, 1), (2, 1, 1)])
def test_supercell_inas(sample, supercell_size):
    """
    Test that the eigenvalues from a supercell model match the folded
    eigenvalues of the base model, for the realistic InAs model.
    """
    model = tbmodels.io.load(sample("InAs_nosym.hdf5"))
    supercell_model = model.supercell(size=supercell_size)
    for k in [(-0.4, 0.1, 0.45), (0, 0, 0), (0.41126, -0.153112, 0.2534)]:
        ev_supercell = supercell_model.eigenval(k)

        equivalent_k = get_equivalent_k(k, supercell_size)
        ev_folded = np.sort(
            np.array([model.eigenval(kval) for kval in equivalent_k]).flatten()
        )
        assert ev_supercell.shape == ev_folded.shape
        assert_allclose(ev_supercell, ev_folded, atol=1e-7)


def test_supercell_model_equal(sample, models_close):
    """
    Regression test checking that a supercell model matches a stored
    reference.
    """
    model = tbmodels.io.load(sample("InAs_nosym.hdf5"))
    supercell_model = model.supercell(size=(1, 2, 3))
    supercell_reference = tbmodels.io.load(sample("InAs_supercell_reference.hdf5"))
    models_close(supercell_model, supercell_reference, ignore_sparsity=True)
