# -*- coding: utf-8 -*-
#
# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import copy

import pytest
import numpy as np

import tbmodels


@pytest.fixture
def model_dense(sample):
    model = tbmodels.io.load(sample('InAs_nosym.hdf5'))
    model.set_sparse(False)
    return model


@pytest.fixture
def model_sparse(sample):
    model = tbmodels.io.load(sample('InAs_nosym.hdf5'))
    model.set_sparse(True)
    return model


@pytest.fixture(params=[True, False])
def model(request, model_dense, model_sparse):
    if request.param:
        return model_sparse
    return model_dense


@pytest.mark.parametrize('num_models', range(1, 4))
def test_join_models(model, num_models):
    model_list = [model] * num_models
    joined_model = tbmodels.Model.join_models(*model_list)

    for k in [[0., 0., 0.], [0.1231, 0.236, 0.84512]]:
        assert np.allclose(sorted(list(model.eigenval(k)) * num_models), joined_model.eigenval(k))


def test_join_mixed_sparsity(model_dense, model_sparse, models_close):
    assert models_close(
        tbmodels.Model.join_models(model_sparse, model_dense), tbmodels.Model.join_models(model_dense, model_sparse)
    )
