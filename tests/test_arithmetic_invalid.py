#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import pytest
import tbmodels
import numpy as np

from parameters import T_VALUES, KPT

T1 = (0.1, 0.2)


def test_add_invalid_type(get_model, compare_equal):
    m = get_model(*T1)
    with pytest.raises(ValueError):
        m2 = m + 2


def test_add_invalid_occ(get_model, compare_equal):
    m1 = get_model(*T1)
    m2 = get_model(*T1, occ=2)
    with pytest.raises(ValueError):
        m3 = m1 + m2


def test_add_invalid_uc(get_model, compare_equal):
    m1 = get_model(*T1, uc=np.eye(3))
    m2 = get_model(*T1, uc=2 * np.eye(3))
    with pytest.raises(ValueError):
        m1 + m2


def test_add_invalid_uc_2(get_model, compare_equal):
    m1 = get_model(*T1, uc=None)
    m2 = get_model(*T1, uc=2 * np.eye(3))
    with pytest.raises(ValueError):
        m1 + m2


def test_add_invalid_nstates(get_model, compare_equal):
    m1 = tbmodels.Model.from_hop_list(size=3, dim=3)
    m2 = tbmodels.Model.from_hop_list(size=4, dim=3)
    with pytest.raises(ValueError):
        m1 + m2


def test_add_invalid_pos(get_model, compare_equal):
    m1 = tbmodels.Model.from_hop_list(size=2, dim=2, pos=((0, 0), (0, 0)))
    m2 = tbmodels.Model.from_hop_list(size=2, dim=2, pos=((0, 0), (0.5, 0.5)))
    with pytest.raises(ValueError):
        m1 + m2


def test_add_invalid_pos_2(get_model, compare_equal):
    m1 = tbmodels.Model.from_hop_list(size=2, dim=2, pos=((0, 0), (0, 0)))
    m2 = tbmodels.Model.from_hop_list(size=2, dim=2, pos=((0.5, 0), (0.5, 0.5)))
    with pytest.raises(ValueError):
        m1 + m2
