#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the method of removing long-range hopping terms."""

import numpy as np


def test_simple_model(get_model):
    """
    Check that removing the long-range hopping in a simple model works
    as expected.
    """
    model = get_model(t1=1, t2=2, uc=np.eye(3))

    # baseline -- check the initial state
    assert set(model.hop.keys()) == set([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)])
    _check_zero(model, (0, 0, 0), [[False, False], [False, False]])
    _check_zero(model, (0, 1, 0), [[False, True], [False, False]])
    _check_zero(model, (1, 0, 0), [[False, True], [False, False]])
    _check_zero(model, (1, 1, 0), [[True, True], [False, True]])

    # check that using a long cut-off does not change anything
    model.remove_long_range_hop(cutoff_distance_cartesian=1.1)
    assert set(model.hop.keys()) == set([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)])
    _check_zero(model, (0, 0, 0), [[False, False], [False, False]])
    _check_zero(model, (0, 1, 0), [[False, True], [False, False]])
    _check_zero(model, (1, 0, 0), [[False, True], [False, False]])
    _check_zero(model, (1, 1, 0), [[True, True], [False, True]])

    # remove next-nearest neighbor hoppings
    model.remove_long_range_hop(cutoff_distance_cartesian=0.9)
    assert set(model.hop.keys()) == set([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)])
    _check_zero(model, (0, 0, 0), [[False, False], [False, False]])
    _check_zero(model, (0, 1, 0), [[True, True], [False, True]])
    _check_zero(model, (1, 0, 0), [[True, True], [False, True]])
    _check_zero(model, (1, 1, 0), [[True, True], [False, True]])

    # remove nearest neighbor hoppings
    model.remove_long_range_hop(cutoff_distance_cartesian=0.5)
    assert set(model.hop.keys()) == set([(0, 0, 0)])
    _check_zero(model, (0, 0, 0), [[False, True], [True, False]])

    # remove everything
    model.remove_long_range_hop(cutoff_distance_cartesian=-1)
    assert set(model.hop.keys()) == set()


def test_model_skewed_uc(get_model):
    """
    Check that removing the long-range hopping in a model with skewed
    unit cell works as expected.
    """
    model = get_model(t1=1, t2=2, uc=np.array([[2, 0, 0], [2, 2, 0], [0, 0, 1]]))

    # baseline -- check the initial state
    assert set(model.hop.keys()) == set([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)])
    _check_zero(model, (0, 0, 0), [[False, False], [False, False]])
    _check_zero(model, (0, 1, 0), [[False, True], [False, False]])
    _check_zero(model, (1, 0, 0), [[False, True], [False, False]])
    _check_zero(model, (1, 1, 0), [[True, True], [False, True]])

    # check that using a long cut-off does not change anything
    model.remove_long_range_hop(cutoff_distance_cartesian=np.sqrt(8) + 0.1)
    assert set(model.hop.keys()) == set([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)])
    _check_zero(model, (0, 0, 0), [[False, False], [False, False]])
    _check_zero(model, (0, 1, 0), [[False, True], [False, False]])
    _check_zero(model, (1, 0, 0), [[False, True], [False, False]])
    _check_zero(model, (1, 1, 0), [[True, True], [False, True]])

    # remove next-nearest neighbor hoppings along a_2 (length sqrt(8))
    model.remove_long_range_hop(cutoff_distance_cartesian=np.sqrt(5) + 0.1)
    assert set(model.hop.keys()) == set([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)])
    _check_zero(model, (0, 0, 0), [[False, False], [False, False]])
    _check_zero(model, (1, 0, 0), [[False, True], [False, False]])
    _check_zero(model, (0, 1, 0), [[True, True], [False, True]])
    _check_zero(model, (1, 1, 0), [[True, True], [False, True]])

    # remove nearest neighbor hoppings along a_1 + a_2 (length sqrt(5))
    model.remove_long_range_hop(cutoff_distance_cartesian=2.1)
    assert set(model.hop.keys()) == set([(0, 0, 0), (0, 1, 0), (1, 0, 0)])
    _check_zero(model, (0, 0, 0), [[False, True], [True, False]])
    _check_zero(model, (1, 0, 0), [[False, True], [False, False]])
    _check_zero(model, (0, 1, 0), [[True, True], [False, True]])

    # remove next-nearest neighbor hoppings along a_1 (length 2)
    model.remove_long_range_hop(cutoff_distance_cartesian=1.5)
    assert set(model.hop.keys()) == set([(0, 0, 0), (0, 1, 0), (1, 0, 0)])
    _check_zero(model, (0, 0, 0), [[False, True], [True, False]])
    _check_zero(model, (1, 0, 0), [[True, True], [False, True]])
    _check_zero(model, (0, 1, 0), [[True, True], [False, True]])

    # remove short nearest-neighbor hopping along a_2 - a_1 (length 1)
    model.remove_long_range_hop(cutoff_distance_cartesian=0.9)
    assert set(model.hop.keys()) == set([(0, 0, 0)])
    _check_zero(model, (0, 0, 0), [[False, True], [True, False]])

    # remove everything
    model.remove_long_range_hop(cutoff_distance_cartesian=-1)
    assert set(model.hop.keys()) == set()


def test_model_skewed_uc_and_pos(get_model):
    """
    Check that removing the long-range hopping in a model with skewed
    unit cell and positions works as expected.
    """
    model = get_model(
        t1=1,
        t2=2,
        uc=np.array([[4, 0, 0], [4, 4, 0], [0, 0, 1]]),
        pos=[[0, 0, 0], [0.5, 0.25, 0]],
    )

    # baseline -- check the initial state
    assert set(model.hop.keys()) == set([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)])
    _check_zero(model, (0, 0, 0), [[False, False], [False, False]])
    _check_zero(model, (0, 1, 0), [[False, True], [False, False]])
    _check_zero(model, (1, 0, 0), [[False, True], [False, False]])
    _check_zero(model, (1, 1, 0), [[True, True], [False, True]])

    # check that using a long cut-off does not change anything
    model.remove_long_range_hop(cutoff_distance_cartesian=np.sqrt(34) + 0.1)
    assert set(model.hop.keys()) == set([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)])
    _check_zero(model, (0, 0, 0), [[False, False], [False, False]])
    _check_zero(model, (0, 1, 0), [[False, True], [False, False]])
    _check_zero(model, (1, 0, 0), [[False, True], [False, False]])
    _check_zero(model, (1, 1, 0), [[True, True], [False, True]])

    # remove longest "nearest-neighbor" hopping - length sqrt(34)
    model.remove_long_range_hop(cutoff_distance_cartesian=np.sqrt(34) - 0.1)
    assert set(model.hop.keys()) == set([(0, 0, 0), (0, 1, 0), (1, 0, 0)])
    _check_zero(model, (0, 0, 0), [[False, False], [False, False]])
    _check_zero(model, (0, 1, 0), [[False, True], [False, False]])
    _check_zero(model, (1, 0, 0), [[False, True], [False, False]])

    # remove same-orbital hoppings along a_2 - length sqrt(32)
    model.remove_long_range_hop(cutoff_distance_cartesian=np.sqrt(32) - 0.1)
    assert set(model.hop.keys()) == set([(0, 0, 0), (0, 1, 0), (1, 0, 0)])
    _check_zero(model, (0, 0, 0), [[False, False], [False, False]])
    _check_zero(model, (1, 0, 0), [[False, True], [False, False]])
    _check_zero(model, (0, 1, 0), [[True, True], [False, True]])

    # remove same-orbital hoppings along a_1 - length 4
    model.remove_long_range_hop(cutoff_distance_cartesian=3.9)
    assert set(model.hop.keys()) == set([(0, 0, 0), (0, 1, 0), (1, 0, 0)])
    _check_zero(model, (0, 0, 0), [[False, False], [False, False]])
    _check_zero(model, (1, 0, 0), [[True, True], [False, True]])
    _check_zero(model, (0, 1, 0), [[True, True], [False, True]])

    # remove two kinds of nearest-neighbor hopping - length sqrt(10)
    model.remove_long_range_hop(cutoff_distance_cartesian=np.sqrt(10) - 0.1)
    assert set(model.hop.keys()) == set([(0, 0, 0), (1, 0, 0)])
    _check_zero(model, (0, 0, 0), [[False, True], [True, False]])
    _check_zero(model, (1, 0, 0), [[True, True], [False, True]])

    # remove last nearest-neighbor hopping - length sqrt(2)
    model.remove_long_range_hop(cutoff_distance_cartesian=np.sqrt(2) - 0.1)
    assert set(model.hop.keys()) == set([(0, 0, 0)])
    _check_zero(model, (0, 0, 0), [[False, True], [True, False]])

    # remove everything
    model.remove_long_range_hop(cutoff_distance_cartesian=-1)
    assert set(model.hop.keys()) == set()


def _check_zero(model, R, expected_zero):
    """
    Helper function to check that the hopping matrix is zero at the
    expected positions.
    """
    assert np.all((np.array(model.hop[R]) == 0) == expected_zero), f"failed at R={R}"
