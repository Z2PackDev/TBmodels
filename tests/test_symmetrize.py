#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests for the 'symmetrize' method.
"""
import copy

import pytest

import tbmodels


@pytest.fixture
def input_model(sample):
    return tbmodels.io.load(sample("InAs_nosym.hdf5"))


@pytest.fixture
def symmetries(sample):
    return tbmodels.io.load(sample("InAs_symmetries.hdf5"))


def test_symmetrize(
    models_close,
    input_model,  # pylint: disable=redefined-outer-name
    symmetries,  # pylint: disable=redefined-outer-name
    sample,
):
    """
    Test the 'symmetrize' method.
    """
    model_res = input_model
    for sym in symmetries:
        if hasattr(sym, "full_group"):
            model_res = model_res.symmetrize(sym.symmetries, full_group=sym.full_group)
        else:
            model_res = model_res.symmetrize([sym], full_group=False)

    model_reference = tbmodels.io.load(sample("InAs_sym_reference.hdf5"))
    models_close(model_res, model_reference)


def test_position_tolerance(
    models_close,
    input_model,  # pylint: disable=redefined-outer-name
    symmetries,  # pylint: disable=redefined-outer-name
    sample,
):
    """
    Test the 'position_tolerance' argument in the 'symmetrize' method.
    """
    model_in = copy.deepcopy(input_model)
    model_reference = tbmodels.io.load(sample("InAs_sym_reference.hdf5"))
    model_in.pos[0] += 0.01
    model_reference.pos[0] += 0.01

    # First run without 'position_tolerance' argument - this should raise
    with pytest.raises(tbmodels.exceptions.TbmodelsException):
        model_res = model_in
        for sym in symmetries:
            if hasattr(sym, "full_group"):
                model_res = model_res.symmetrize(
                    sym.symmetries, full_group=sym.full_group
                )
            else:
                model_res = model_res.symmetrize([sym], full_group=False)

    # Adding the 'position_tolerance' argument suppresses the error
    model_res = model_in
    for sym in symmetries:
        if hasattr(sym, "full_group"):
            model_res = model_res.symmetrize(
                sym.symmetries, full_group=sym.full_group, position_tolerance=0.05
            )
        else:
            model_res = model_res.symmetrize(
                [sym], full_group=False, position_tolerance=0.05
            )

    models_close(model_res, model_reference)
