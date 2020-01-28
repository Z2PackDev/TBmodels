#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""Tests for the model slicing functionality."""

import pytest
import numpy as np

from parameters import T_VALUES, KPT


@pytest.mark.parametrize('slice_idx', [(0, 1), [1, 0], (0, ), (1, )])
@pytest.mark.parametrize('t', T_VALUES)
def test_slice(t, get_model, slice_idx):
    """Check the slicing method."""
    model = get_model(*t)
    model_sliced = model.slice_orbitals(slice_idx)
    assert np.isclose([model.pos[i] for i in slice_idx], model_sliced.pos).all()
    for k in KPT:
        assert np.isclose(
            model.hamilton(k)[np.ix_(slice_idx, slice_idx)], model_sliced.hamilton(k)
        ).all()
