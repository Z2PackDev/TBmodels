#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for constructing supercell models."""

import pytest

import tbmodels

from parameters import T_VALUES


@pytest.mark.parametrize('t_values', T_VALUES)
@pytest.mark.parametrize('offset', [(0.2, 1.2, 0.9), (-0.2, 0.912, 0.)])
@pytest.mark.parametrize('cartesian', [True, False])
def test_shift_twice(get_model, t_values, sparse, offset, cartesian, models_close):
    """
    Check that shifting a model twice in opposite direction gives back
    the original model.
    """
    model = get_model(*t_values, sparse=sparse, uc=[[0.1, 1., 0.], [2., 0., 0.], [0., 0., 3.]])
    model_shifted = model.shift_unit_cell(offset=offset, cartesian=cartesian)
    model_shifted_twice = model_shifted.shift_unit_cell(
        offset=[-x for x in offset], cartesian=cartesian
    )
    assert models_close(model, model_shifted_twice)


def test_shift_supercell(sample, models_close):
    """
    Test that shifting the unit cell is equivalent to creating and
    then folding a supercell.
    """
    model = tbmodels.io.load(sample('InAs_nosym.hdf5'))
    shift = model.uc[2] / 2 - 0.03

    model_shifted = model.shift_unit_cell(offset=shift, cartesian=True)

    supercell_model = model.supercell(size=(1, 1, 2))
    model_folded = supercell_model.fold_model(
        new_unit_cell=model.uc, unit_cell_offset=shift, orbital_labels=list(range(model.size)) * 2
    )

    assert models_close(model_shifted, model_folded)
