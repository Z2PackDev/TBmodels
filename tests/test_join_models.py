# -*- coding: utf-8 -*-
#
# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""Tests for joining two models together."""

import pytest
import numpy as np

import tbmodels


@pytest.fixture
def model_dense(sample):
    """Fixture for a dense tight-binding model."""
    res = tbmodels.io.load(sample('InAs_nosym.hdf5'))
    res.set_sparse(False)
    return res


@pytest.fixture
def model_sparse(sample):
    """Fixture for a sparse tight-binding model."""
    res = tbmodels.io.load(sample('InAs_nosym.hdf5'))
    res.set_sparse(True)
    return res


@pytest.fixture
def model(model_dense, model_sparse, sparse):  # pylint: disable=redefined-outer-name
    """Fixture to get a tight-binding model, both sparse and dense."""
    if sparse:
        return model_sparse
    return model_dense


@pytest.mark.parametrize('num_models', range(1, 4))
def test_join_models(model, num_models):  # pylint: disable=redefined-outer-name
    """Test joining equal models."""
    model_list = [model] * num_models
    joined_model = tbmodels.Model.join_models(*model_list)

    for k in [[0., 0., 0.], [0.1231, 0.236, 0.84512]]:
        assert np.allclose(sorted(list(model.eigenval(k)) * num_models), joined_model.eigenval(k))


def test_join_mixed_sparsity(model_dense, model_sparse, models_close):  # pylint: disable=redefined-outer-name
    """Test joining dense and sparse models."""
    assert models_close(
        tbmodels.Model.join_models(model_sparse, model_dense),
        tbmodels.Model.join_models(model_dense, model_sparse)
    )
