#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""Tests for model sparsity handling / switching."""

import pytest
import numpy as np

from parameters import T_VALUES, KPT

import tbmodels


@pytest.mark.parametrize("t1", T_VALUES)
def test_simple(t1, get_model):
    """
    Check that a simple model set up as sparse and dense creates
    the same Hamiltonians.
    """
    model_dense = get_model(*t1, sparse=True)
    model_sparse = get_model(*t1, sparse=False)

    for k in KPT:
        assert np.isclose(model_dense.hamilton(k), model_sparse.hamilton(k)).all()


@pytest.mark.parametrize("t1", T_VALUES)
def test_change_to_dense(t1, get_model, models_close):
    """
    Check that creating a sparse model and then switching it to dense
    creates the same result as directly creating a dense model.
    """
    model1 = get_model(*t1, sparse=True)
    model2 = get_model(*t1, sparse=False)
    model1.set_sparse(False)
    assert models_close(model1, model2)


@pytest.mark.parametrize("t1", T_VALUES)
def test_change_to_sparse(t1, get_model, models_close):
    """
    Check that creating a dense model and then switching it to sparse
    creates the same result as directly creating a sparse model.
    """
    model1 = get_model(*t1, sparse=True)
    model2 = get_model(*t1, sparse=False)
    model2.set_sparse(True)
    assert models_close(model1, model2)


@pytest.mark.parametrize(
    "hr_name", ["hr_hamilton.dat", "wannier90_hr.dat", "wannier90_hr_v2.dat"]
)
def test_hr(hr_name, sample):
    """
    Check that models loaded from *_hr.dat format have the same Hamiltonians
    when loaded as either sparse or dense models.
    """
    hr_file = sample(hr_name)
    model1 = tbmodels.Model.from_wannier_files(hr_file=hr_file, occ=28, sparse=False)
    model2 = tbmodels.Model.from_wannier_files(hr_file=hr_file, occ=28, sparse=True)

    for k in KPT:
        assert np.isclose(model1.hamilton(k), model2.hamilton(k)).all()
