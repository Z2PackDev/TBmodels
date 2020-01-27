#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for folding models."""

import numpy as np
import pytest

from parameters import T_VALUES


@pytest.mark.parametrize('t_values', T_VALUES)
@pytest.mark.parametrize('supercell_size', [(1, 1, 1), (2, 1, 1), (2, 1, 3)])
def test_fold_supercell_simple(get_model, t_values, supercell_size, sparse, models_close):
    """
    Test that creating a supercell model and then folding it back creates
    the same model.
    """
    model = get_model(*t_values, sparse=sparse)
    model.uc = np.array([[1, 0, 0.5], [0.1, 0.4, 0.], [0., 0., 1.2]])
    model.pos = np.array([[0, 0, 0], [0.2, 0.3, 0.1]])
    supercell_model = model.supercell(size=supercell_size)
    orbital_labels = ['a', 'b'] * np.prod(supercell_size)
    for i, offset_red in enumerate(supercell_model.pos[::2]):
        # TODO: changing the 'offset_red' manually is a temporary fix:
        # to be removed when more complex orbital matching is implemented.
        offset_cart = supercell_model.uc.T @ (offset_red - 1e-12)
        folded_model = supercell_model.fold_model(
            new_unit_cell=model.uc,
            unit_cell_offset=offset_cart,
            orbital_labels=orbital_labels,
            target_indices=[2 * i, 2 * i + 1]
        )
        assert models_close(model, folded_model)


def test_fold_inexact_positions(get_model, models_close):
    """
    Test that creating a supercell model and then folding it back creates
    the same model.
    """
    model = get_model(0.1, 0.3)
    model.uc = np.array([[1, 0, 0.5], [0.1, 0.4, 0.], [0., 0., 1.2]])
    model.pos = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    supercell_model = model.supercell(size=(8, 1, 1))
    orbital_labels = ['a', 'b'] * 8
    np.random.seed(42)
    for i in range(len(supercell_model.pos)):
        # do not move positions around "base" unit cell
        if i in range(3, 7):
            continue
        delta = np.random.uniform(-0.01, 0.01, 3)
        supercell_model.pos[i] += delta
    folded_model = supercell_model.fold_model(
        new_unit_cell=model.uc,
        unit_cell_offset=supercell_model.uc.T @ supercell_model.pos[4],
        orbital_labels=orbital_labels,
        position_tolerance=0.1,
        check_cc=False
    )
    assert models_close(model, folded_model)
