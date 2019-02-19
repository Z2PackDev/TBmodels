#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import pytest
import numpy as np

import tbmodels
from tbmodels._kdotp import KdotpModel
from parameters import KPT


def test_kdotp_model():
    model = KdotpModel({(0, 0): np.eye(2), (1, 0): [[0, 1j], [-1j, 0]], (0, 2): [[0, 1], [1, 0]]})

    assert np.allclose(model.hamilton((0, 0)), np.eye(2))
    assert np.allclose(model.hamilton((1, 0)), [[1, 1j], [-1j, 1]])
    assert np.allclose(model.hamilton((0, 0.5)), [[1, 0.25], [0.25, 1]])

    assert np.allclose(model.eigenval((0, 0)), [1, 1])


def test_raises_not_hermitian():
    with pytest.raises(ValueError):
        KdotpModel({(0, 0): [[0, 1], [2, 0]]})


@pytest.mark.parametrize('kpt', KPT)
@pytest.mark.parametrize('order', [0, 1, 2, 3])
def test_construct_kdotp(sample, kpt, order):
    model_tb = tbmodels.io.load(sample('InAs_nosym.hdf5'))
    model_kp = model_tb.construct_kdotp(kpt, order=order)

    assert np.allclose(model_kp.eigenval((0, 0, 0)), model_tb.eigenval(kpt))


def test_construct_kdotp_negative_order(sample):
    model_tb = tbmodels.io.load(sample('InAs_nosym.hdf5'))
    with pytest.raises(ValueError):
        model_tb.construct_kdotp((0, 0, 0), order=-1)
