#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""Configuration file for pytest tests."""

# pylint: disable=redefined-outer-name

import os
import operator
import itertools
from functools import partial
from collections import ChainMap

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal

import tbmodels
from tbmodels.io import save, load

#--------------------------FIXTURES-------------------------------------#


@pytest.fixture
def test_name(request):
    """Returns (module_name, function_name[args]) for a given test"""
    return (request.module.__name__, request._parent_request._pyfuncitem.name)  # pylint: disable=protected-access


@pytest.fixture
def compare_data(test_name):
    """Returns a function which either saves some data to a file or (if that file exists already) compares it to pre-existing data using a given comparison function."""
    def inner(compare_fct, data, tag=None):
        dir_name, file_name = test_name
        file_name += tag or ''
        cache_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'regression_data', dir_name
        )
        os.makedirs(cache_dir, exist_ok=True)
        file_name_full = os.path.join(cache_dir, file_name)
        try:
            val = load(file_name_full)
        except OSError:
            save(data, file_name_full)
            raise ValueError('Reference data does not exist.')
        else:
            assert compare_fct(val, data)

    return inner


@pytest.fixture
def compare_equal(compare_data):
    return lambda data, tag=None: compare_data(operator.eq, data, tag)


@pytest.fixture
def compare_isclose(compare_data):
    return lambda data, tag=None: compare_data(np.allclose, data, tag)


@pytest.fixture
def models_equal():
    """
    Check that two tight-binding models are equal.
    """
    def inner(model1, model2, ignore_sparsity=False):
        assert model1.size == model2.size
        assert model1.dim == model2.dim
        assert np.array(model1.uc == model2.uc).all()
        assert model1.occ == model2.occ
        for k in model1.hop.keys() | model2.hop.keys():
            assert (np.array(model1.hop[k]) == np.array(model2.hop[k])).all(
            ), f"Hoppings unequal at k={k}\nmodel1:\n{np.array(model1.hop[k])}\nmodel2:\n{np.array(model2.hop[k])}"
        assert_equal(model1.pos, model2.pos)
        if not ignore_sparsity:
            assert model1._sparse == model2._sparse, "Sparsity does not match"  # pylint: disable=protected-access
        return True

    return inner


@pytest.fixture
def kdotp_models_equal():
    """
    Check that two k.p models are equal.
    """
    def inner(model1, model2):
        for power in model1.taylor_coefficients.keys() | model1.taylor_coefficients.keys():
            assert (
                np.array(model1.taylor_coefficients[power]
                         ) == np.array(model2.taylor_coefficients[power])
            ).all()
        return True

    return inner


@pytest.fixture
def models_close():
    """
    Check that two tight-binding models are almost equal.
    """
    def inner(model1, model2, ignore_sparsity=False):
        assert model1.size == model2.size
        assert model1.dim == model2.dim
        if model1.uc is None:
            assert model1.uc == model2.uc
        else:
            assert_allclose(model1.uc, model2.uc)
        assert model1.occ == model2.occ
        if model1.pos is None:
            assert model1.pos == model2.pos
        else:
            assert_allclose(model1.pos, model2.pos, atol=1e-7)
        for k in model1.hop.keys() | model2.hop.keys():
            assert np.isclose(np.array(model1.hop[k]), np.array(model2.hop[k])).all(
            ), f"Hoppings unequal at k={k}\nmodel1:\n{np.array(model1.hop[k])}\nmodel2:\n{np.array(model2.hop[k])}"

        if not ignore_sparsity:
            assert model1._sparse == model2._sparse  # pylint: disable=protected-access
        return True

    return inner


@pytest.fixture
def sample():
    """
    Get the absolute path of a file / directory in the 'samples' directory.
    """
    def inner(name):
        return os.path.join(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'samples'), name
        )

    return inner


#-----------------------------------------------------------------------#
@pytest.fixture
def get_model_clean():
    """
    Function that creates a simple tight-binding model
    """
    def inner(t1, t2, sparsity_default=False, **kwargs):
        dim = kwargs.get('dim', 3)
        defaults = {}
        defaults['pos'] = [[0] * 2, [0.5] * 2]
        if dim < 2:
            raise ValueError('dimension must be at least 2')
        if dim > 2:
            for position in defaults['pos']:
                position.extend([0] * (dim - 2))
        defaults['occ'] = 1
        defaults['on_site'] = (1, -1)
        defaults['size'] = 2
        defaults['dim'] = None
        defaults['sparse'] = sparsity_default
        model = tbmodels.Model(**ChainMap(kwargs, defaults))

        for phase, r_part in zip([1, -1j, 1j, -1], itertools.product([0, -1], [0, -1])):
            R = list(r_part)
            R.extend([0] * (dim - 2))
            model.add_hop(t1 * phase, 0, 1, R)

        for r_part in itertools.permutations([0, 1]):
            R = list(r_part)
            R.extend([0] * (dim - 2))
            model.add_hop(t2, 0, 0, R)
            model.add_hop(-t2, 1, 1, R)
        return model

    return inner


@pytest.fixture(params=[True, False])
def sparse(request):
    """
    Fixture to set the sparsity to either True or False.
    """
    return request.param


@pytest.fixture()  # params is for sparse / dense
def get_model(get_model_clean, sparse):
    """
    Function that creates a simple tight-binding model, with default
    sparsity determined by the 'sparse' fixture.
    """
    return partial(get_model_clean, sparsity_default=sparse)


@pytest.fixture(params=[None, 'as_input', 'sparse', 'dense'])
def cli_sparsity(request):
    return request.param


@pytest.fixture
def cli_sparsity_arguments(cli_sparsity):
    if cli_sparsity is None:
        return []
    return ['--sparsity', cli_sparsity]


@pytest.fixture(params=[False, True])
def cli_verbosity_argument(request):
    if request.param:
        return ['--verbose']
    return []


@pytest.fixture
def modify_reference_model_sparsity(cli_sparsity):
    """
    Modify the sparsity of the given reference model in-place to match
    the value given in the sparsity argument.
    """
    def _modify_model(model):
        if cli_sparsity == 'dense':
            model.set_sparse(False)
        elif cli_sparsity == 'sparse':
            model.set_sparse(True)

    return _modify_model
