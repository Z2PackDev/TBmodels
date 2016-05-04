#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    19.04.2016 11:55:35 CEST
# File:    test_uc_change.py

import pytest
import numpy as np

import tbmodels


@pytest.fixture
def get_model():
    def inner(t1, t2, dim=3, uc=None, pos=None):
        model = tbmodels.Model(
            on_site=[1, -1, 0],
            pos=[
                [0, 0., 0.],
                [0.5, 0.5, 0.2],
                [0.75, 0.15, 0.6],
            ],
            occ=1,
            uc=uc,
            dim=3
        )

        for phase, G in zip([1, -1j, 1j, -1], tbmodels.helpers.combine([0, -1], [0, -1], 0)):
            model.add_hop(t1 * phase, 0, 1, G)

        for G in tbmodels.helpers.neighbours([0, 1], forward_only=True):
            model.add_hop(t2, 0, 0, G)
            model.add_hop(-t2, 1, 1, G)

        return model
    return inner


def test_no_uc_unity(get_model):
    model = get_model(0.1, 0.2)
    new_model = model.change_uc(uc=np.identity(3))
    assert new_model.uc == model.uc
    assert np.isclose(model.pos, new_model.pos).all()


def test_uc_unity(get_model):
    model = get_model(0.1, 0.2, uc=np.diag([1, 2, 3]))
    new_model = model.change_uc(uc=np.identity(3))
    assert np.isclose(new_model.uc, model.uc).all()
    assert np.isclose(model.pos, new_model.pos).all()


def test_uc_1(get_model):
    model = get_model(0.1, 0.2, uc=np.diag([1, 2, 3]))
    new_model = model.change_uc(uc=np.array([[1, 2, 0], [0, 1, 3], [0, 0, 1]]))

    res_uc = np.array([
        [1, 2, 0],
        [0, 2, 6],
        [0, 0, 3]
    ])
    res_pos = np.array([
        [0., 0., 0.],
        [0.7, 0.9, 0.2],
        [0.05, 0.35, 0.6]
    ])
    assert np.isclose(new_model.uc, res_uc).all()
    assert np.isclose(new_model.pos, res_pos).all()


def test_uc_2(get_model):
    model = get_model(0.1, 0.2, uc=np.array([[0, 1, 0], [2, 0, 1], [1, 0, 1]]))
    new_model = model.change_uc(uc=np.array([[1, 2, 0], [0, 1, 3], [0, 0, 1]]))

    res_uc = np.array([
        [0, 1, 3],
        [2, 4, 1],
        [1, 2, 1]
    ])
    res_pos = np.array([
        [0., 0., 0.],
        [0.7, 0.9, 0.2],
        [0.05, 0.35, 0.6]
    ])
    assert np.isclose(new_model.uc, res_uc).all()
    assert np.isclose(new_model.pos, res_pos).all()


def test_uc_hamilton_unity(get_model):
    model = get_model(0.1, 0.2, uc=np.diag([1, 2, 3]))
    new_model = model.change_uc(uc=np.identity(3))

    res = np.array([
        [1.0 + 0.j,  0.2 + 0.1618034j,  0.0 + 0.j],
        [0.2 - 0.1618034j, -1.0 + 0.j,  0.0 + 0.j],
        [0.0 + 0.j,  0.0 + 0.j,  0.0 + 0.j]
    ])
    assert np.isclose(res, new_model.hamilton([0.1, 0.4, 0.7])).all()


def test_uc_hamilton_1(get_model):
    model = get_model(0.1, 0.2, uc=np.diag([1, 2, 3]))
    new_model = model.change_uc(uc=np.array([[1, 2, 0], [0, 1, 3], [0, 0, 1]]))

    res = np.array([
        [1.44721360 + 0.j,  0.00877853 - 0.17298248j,  0.00000000 + 0.j],
        [0.00877853 + 0.17298248j, -1.44721360 + 0.j,  0.00000000 + 0.j],
        [0.00000000 + 0.j,  0.00000000 + 0.j,  0.00000000 + 0.j]
    ])
    assert np.isclose(res, new_model.hamilton([0.1, 0.4, 0.7])).all()


def test_uc_hamilton_2(get_model):
    model = get_model(0.1, 0.2, uc=np.array([[0, 1, 0], [2, 0, 1], [1, 0, 1]]))
    new_model = model.change_uc(uc=np.array([[1, 2, 0], [0, 1, 3], [0, 0, 1]]))

    res = np.array([
        [1.44721360 + 0.j,  0.00877853 - 0.17298248j,  0.00000000 + 0.j],
        [0.00877853 + 0.17298248j, -1.44721360 + 0.j,  0.00000000 + 0.j],
        [0.00000000 + 0.j,  0.00000000 + 0.j,  0.00000000 + 0.j]
    ])
    assert np.isclose(res, new_model.hamilton([0.1, 0.4, 0.7])).all()


def test_error(get_model):
    model = get_model(0.1, 0.2, uc=np.identity(3))
    with pytest.raises(ValueError):
        model.change_uc(uc=np.diag([2, 1, 1]))
