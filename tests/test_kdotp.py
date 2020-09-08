#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Test the construction of k.p models around a given k-point.
"""

import pytest
import numpy as np

from parameters import KPT
import tbmodels
from tbmodels.kdotp import KdotpModel


def test_kdotp_model():
    """Check that a k.p model can be explicitly created."""
    model = KdotpModel(
        {(0, 0): np.eye(2), (1, 0): [[0, 1j], [-1j, 0]], (0, 2): [[0, 1], [1, 0]]}
    )

    assert np.allclose(model.hamilton((0, 0)), np.eye(2))
    assert np.allclose(model.hamilton((1, 0)), [[1, 1j], [-1j, 1]])
    assert np.allclose(model.hamilton((0, 0.5)), [[1, 0.25], [0.25, 1]])
    assert np.allclose(
        model.hamilton([(0, 0), (0, 0.5), (1, 0)]),
        [np.eye(2), [[1, 0.25], [0.25, 1]], [[1, 1j], [-1j, 1]]],
    )

    assert np.allclose(model.eigenval((0, 0)), [1, 1])
    assert np.allclose(model.eigenval([(0, 0), (0, 0)]), [[1, 1], [1, 1]])
    assert np.allclose(
        model.eigenval([(0, 0), (0, 0), (0, 0.5)]), [[1, 1], [1, 1], [0.75, 1.25]]
    )


def test_raises_not_hermitian():
    """
    Test that the k.p model constructor raises when trying to construct
    model with a non-hermitian Hamiltonian.
    """
    with pytest.raises(ValueError):
        KdotpModel({(0, 0): [[0, 1], [2, 0]]})


@pytest.mark.parametrize("kpt", KPT)
@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_construct_kdotp(sample, kpt, order):
    """
    Test constructing a k.p model around a given k-point of an InAs
    tight-binding model.
    """
    model_tb = tbmodels.io.load(sample("InAs_nosym.hdf5"))
    model_kp = model_tb.construct_kdotp(kpt, order=order)

    assert np.allclose(model_kp.eigenval((0, 0, 0)), model_tb.eigenval(kpt))


def test_construct_kdotp_negative_order(sample):  # pylint: disable=invalid-name
    """
    Check that passing negative values for the 'order' in k.p construction
    raises an error.
    """
    model_tb = tbmodels.io.load(sample("InAs_nosym.hdf5"))
    with pytest.raises(ValueError):
        model_tb.construct_kdotp((0, 0, 0), order=-1)
