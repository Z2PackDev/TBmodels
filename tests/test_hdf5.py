#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tempfile

import pytest
import tbmodels
import numpy as np

kpt = [(0.1, 0.2, 0.7), (-0.3, 0.5, 0.2), (0., 0., 0.), (0.1, -0.9, -0.7)]

KWARGS = [
    dict(),
    dict(pos=None, dim=3),
    dict(uc=3 * np.eye(3)),
    dict(pos=np.zeros((2, 3)), uc=np.eye(3))
]


@pytest.mark.parametrize('kwargs', KWARGS)
def test_hdf5_consistency_file(get_model, models_equal, kwargs):
    model1 = get_model(0.1, 0.2, **kwargs)
    with tempfile.NamedTemporaryFile() as tmpf:
        model1.to_hdf5_file(tmpf.name)
        model2 = tbmodels.Model.from_hdf5_file(tmpf.name)
    models_equal(model1, model2)


@pytest.fixture(params=['InAs_nosym.hdf5'])
def hdf5_sample(sample, request):
    return sample(request.param)


def test_hdf5_load_freefunc(hdf5_sample):
    res = tbmodels.io.load(hdf5_sample)
    assert isinstance(res, tbmodels.Model)


def test_hdf5_load_method(hdf5_sample):
    res = tbmodels.Model.from_hdf5_file(hdf5_sample)
    assert isinstance(res, tbmodels.Model)
