#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests checking that errors are raised when performing invalid
arithmetic operations on tight-binding models.
"""

# pylint: disable=pointless-statement

import pytest
import numpy as np

import tbmodels

T1 = (0.1, 0.2)


def test_add_invalid_type(get_model):
    model = get_model(*T1)
    with pytest.raises(ValueError):
        model + 2


def test_add_invalid_occ(get_model):
    model1 = get_model(*T1)
    model2 = get_model(*T1, occ=2)
    with pytest.raises(ValueError):
        model1 + model2


def test_add_invalid_uc(get_model):
    model1 = get_model(*T1, uc=np.eye(3))
    model2 = get_model(*T1, uc=2 * np.eye(3))
    with pytest.raises(ValueError):
        model1 + model2


def test_add_invalid_uc_2(get_model):
    model1 = get_model(*T1, uc=None)
    model2 = get_model(*T1, uc=2 * np.eye(3))
    with pytest.raises(ValueError):
        model1 + model2


def test_add_invalid_nstates():
    model1 = tbmodels.Model.from_hop_list(size=3, dim=3)
    model2 = tbmodels.Model.from_hop_list(size=4, dim=3)
    with pytest.raises(ValueError):
        model1 + model2


def test_add_invalid_pos():
    model1 = tbmodels.Model.from_hop_list(size=2, dim=2, pos=((0, 0), (0, 0)))
    model2 = tbmodels.Model.from_hop_list(size=2, dim=2, pos=((0, 0), (0.5, 0.5)))
    with pytest.raises(ValueError):
        model1 + model2


def test_add_invalid_pos_2():
    model1 = tbmodels.Model.from_hop_list(size=2, dim=2, pos=((0, 0), (0, 0)))
    model2 = tbmodels.Model.from_hop_list(size=2, dim=2, pos=((0.5, 0), (0.5, 0.5)))
    with pytest.raises(ValueError):
        model1 + model2
