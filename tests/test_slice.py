#!/usr/bin/env python
# -*- coding: utf-8 -*-

# © 2015-2018, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import pytest
import numpy as np

from parameters import T_VALUES, KPT


@pytest.mark.parametrize('slice_idx', [(0, 1), [1, 0], (0, ), (1, )])
@pytest.mark.parametrize('t', T_VALUES)
def test_slice(t, get_model, slice_idx):
    m1 = get_model(*t)
    m2 = m1.slice_orbitals(slice_idx)
    assert np.isclose([m1.pos[i] for i in slice_idx], m2.pos).all()
    for k in KPT:
        assert np.isclose(m1.hamilton(k)[np.ix_(slice_idx, slice_idx)], m2.hamilton(k)).all()
