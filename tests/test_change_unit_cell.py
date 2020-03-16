#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for constructing supercell models."""

import itertools

import pytest
import numpy as np

from parameters import T_VALUES

# pylint: disable=invalid-name


@pytest.mark.parametrize("t_values", T_VALUES)
@pytest.mark.parametrize("offset", [(0.2, 1.2, 0.9), (-0.2, 0.912, 0.0)])
@pytest.mark.parametrize("cartesian", [True, False])
def test_shift_twice(get_model, t_values, sparse, offset, cartesian, models_close):
    """
    Check that shifting a model twice in opposite direction gives back
    the original model.
    """
    model = get_model(
        *t_values, sparse=sparse, uc=[[0.1, 1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0]]
    )
    model_shifted = model.change_unit_cell(offset=offset, cartesian=cartesian)
    model_shifted_twice = model_shifted.change_unit_cell(
        offset=[-x for x in offset], cartesian=cartesian
    )
    assert models_close(model, model_shifted_twice)


@pytest.mark.parametrize(
    "uc", [((1.1, 0.3, 0.0), (0.4, 1.5, 0.1), (-0.1, 0.0, 3.0)), None]
)
@pytest.mark.parametrize("offset", [(1, 0, -1), (2, 3, 0)])
def test_lattice_shift_reduced(get_model, sparse, offset, uc, models_equal):
    """
    Check that shifting by a lattice vector produces the original
    model, with reduced coordinates.
    """
    model = get_model(t1=0.1, t2=0.7, sparse=sparse, uc=uc)
    model_shifted = model.change_unit_cell(offset=offset, cartesian=False)
    assert models_equal(model, model_shifted)


@pytest.mark.parametrize(
    "uc, offsets",
    [
        (
            [[1.1, 0.3, 0.0], [0.4, 1.5, 0.1], [-0.1, 0.0, 3.0]],
            [(1.1, 0.3, 0), [1.5, 1.8, 0.1], (0.1, 0, -3)],
        )
    ],
)
def test_lattice_shift_cartesian(get_model, sparse, uc, offsets, models_close):
    """
    Check that shifting by a lattice vector produces the original
    model, with cartesian coordinates.
    """
    # We change the position from being exactly at the unit cell boundary
    # to avoid issues with positions being off by one unit cell.
    model = get_model(
        t1=0.1, t2=0.7, sparse=sparse, uc=uc, pos=[(0.01, 0.02, 0.03), (0.5, 0.5, 0.5)]
    )
    for offset in offsets:
        model_shifted = model.change_unit_cell(offset=offset, cartesian=True)
        assert models_close(model, model_shifted)


@pytest.mark.parametrize(
    "uc",
    (
        [[1, 2, 0], [1, 1, 0], [0, 0, 1]],
        [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1.5, 0], [0, 0, 1]],
    ),
)
def test_invalid_uc_raises_reduced(get_model, uc, sparse):
    """
    Test that specifying an invalid new unit cell in reduced coordinates
    raises an error.
    """
    model = get_model(t1=0.1, t2=0.7, sparse=sparse, uc=uc)
    with pytest.raises(ValueError):
        model.change_unit_cell(uc=uc, cartesian=False)


@pytest.mark.parametrize(
    "uc", ([[1, 2, 0], [0, 1, 0], [0, 0, 6]], [[1, 2, 0], [0, 1.5, 0], [0, 0, 3]])
)
def test_invalid_uc_raises_cartesian(get_model, uc, sparse):
    """
    Test that specifying an invalid new unit cell in cartesian coordinates
    raises an error.
    """
    model = get_model(
        t1=0.1, t2=0.7, sparse=sparse, uc=[[1, 2, 0], [0, 1, 0], [0, 0, 3]]
    )
    with pytest.raises(ValueError):
        model.change_unit_cell(uc=uc, cartesian=True)


def test_change_uc_without_pos_raises(get_model):
    """
    Test that the 'change_unit_cell' method raises an error when no
    positions are defined.
    """
    model = get_model(t1=0.4, t2=0.9)
    model.pos = None
    with pytest.raises(ValueError):
        model.change_unit_cell()


def test_change_uc_without_uc_cartesian_raises(get_model):
    """
    Test that the 'change_unit_cell' method raises an error in cartesian
    mode when the original unit cell is not defined.
    """
    model = get_model(t1=0.4, t2=0.9)
    model.uc = None
    with pytest.raises(ValueError):
        model.change_unit_cell(cartesian=True)


@pytest.mark.parametrize(
    "uc_original, uc_changed, offset",
    [
        (
            [[1.2, 0.1, 0.0], [0, 2, 0], [1, 0, 3]],
            [[1.2, 2.1, 0.0], [-1, 2, -3], [1, 0, 3]],
            (0, 0, 0),
        ),
        (
            [[1.2, 0.1, 0.0], [0, 2, 0], [1, 0, 3]],
            [[1.2, 2.1, 0.0], [-1, 2, -3], [1, 0, 3]],
            (0.5, -0.1, 10.2),
        ),
        ([[1.2, 0.1], [0, 2]], [[1.2, 2.1], [0, 2]], (0.2, -1.5)),
    ],
)
def test_revert_cartesian_uc_change(
    get_model, models_close, uc_original, uc_changed, offset
):
    """
    Test that reverting a cartesian unit cell change produces the original model.
    """
    if offset is None:
        revert_offset = None
    else:
        revert_offset = -np.array(offset)
    dim = len(offset)
    model = get_model(
        t1=0.2, t2=0.3, pos=[[0.1] * dim, [0.7] * dim], uc=uc_original, dim=dim
    )
    model_changed = model.change_unit_cell(uc=uc_changed, cartesian=True, offset=offset)
    model_change_reverted = model_changed.change_unit_cell(
        uc=uc_original, cartesian=True, offset=revert_offset
    )
    models_close(model, model_change_reverted)


def test_equivalent_uc_shape(get_model, models_close):
    """
    Test that two manually created equivalent models are equal after
    matching unit cells.
    """
    t1 = 0.232
    t2 = -0.941234
    uc1 = np.eye(3)
    uc2 = [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
    model1 = get_model(t1=0, t2=0, pos=[(0.2, 0.1, 0.1), (0.6, 0.5, 0.5)], uc=uc1)
    model2 = get_model(t1=0, t2=0, pos=[(0.1, 0.1, 0.1), (0.1, 0.5, 0.5)], uc=uc2)

    for phase, R1 in zip([1, -1j, 1j, -1], itertools.product([0, -1], [0, -1], [0])):
        model1.add_hop(t1 * phase, 0, 1, R1)
        R2 = [R1[0] - R1[1], R1[1], R1[2]]
        model2.add_hop(t1 * phase, 0, 1, R2)

    for r1_part in itertools.permutations([0, 1]):
        R1 = list(r1_part) + [0]
        R2 = [R1[0] - R1[1], R1[1], R1[2]]
        model1.add_hop(t2, 0, 0, R1)
        model1.add_hop(-t2, 1, 1, R1)

        model2.add_hop(t2, 0, 0, R2)
        model2.add_hop(-t2, 1, 1, R2)

    assert models_close(model1, model2.change_unit_cell(uc=uc1, cartesian=True))
    assert models_close(model2, model1.change_unit_cell(uc=uc2, cartesian=True))
    assert models_close(model2, model1.change_unit_cell(uc=uc2, cartesian=False))
    assert models_close(
        model1,
        model2.change_unit_cell(uc=[[1, 0, 0], [-1, 1, 0], [0, 0, 1]], cartesian=False),
    )
