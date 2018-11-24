#!/usr/bin/env python

import pytest
import numpy as np

from tbmodels._kdotp import KdotpModel


def test_kdotp_model():
    model = KdotpModel({(0, 0): np.eye(2), (1, 0): [[0, 1j], [-1j, 0]], (0, 2): [[0, 1], [1, 0]]})

    assert np.allclose(model.hamilton((0, 0)), np.eye(2))
    assert np.allclose(model.hamilton((1, 0)), [[1, 1j], [-1j, 1]])
    assert np.allclose(model.hamilton((0, 0.5)), [[1, 0.25], [0.25, 1]])

    assert np.allclose(model.eigenval((0, 0)), [1, 1])


def test_raises_not_hermitian():
    with pytest.raises(ValueError):
        KdotpModel({(0, 0): [[0, 1], [2, 0]]})
