#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    04.06.2015 14:26:21 CEST
# File:    helpers.py

"""
Helper functions for creating tight-binding models.
"""

import numpy as np

def matrix_to_hoppings(mat, orbitals=None, G=[0, 0, 0], multiplier=1.):
    r"""
    Turns a square matrix into a series of hopping terms.

    :param mat: The matrix to be converted.

    :param orbitals:    Indices of the orbitals that make up the basis w.r.t. which the matrix is defined. By default (``orbitals=None``), the first ``len(mat)`` orbitals are used.
    :type orbitals:     list

    :param G:   Reciprocal lattice vector for all the hopping terms.
    :type G:    list

    :param multiplier:  Multiplicative constant for the hoppings strength.
    :type multiplier: float / complex
    """
    if orbitals is None:
        orbitals = list(range(len(mat)))
    hop = []
    for i, row in enumerate(mat):
        for j, x in enumerate(row):
            hop.append([orbitals[i], orbitals[j], np.array(G, dtype=int), multiplier * x])
    return hop


def neighbours(axes, forward_only=True):
    """
    Adds vectors for every axis, either two (with +-1) or one (with +1, \
    default) in that axis (0 on
    the other coordinates).

    :param axes:            Axes for which neighbours are to be added, \
    either as different arguments or as a list
    :type args:             int or list(int)
    :param forward_only:    If True, adds only the neighbour in positive \
    direction (+1) instead of both directions (+-1) ``Default: True``
    :type forward_only:     Boolean
    """
    res = []
    if isinstance(axes, int):
        axes = [axes]

    for axis in axes:
        if not isinstance(axis, int):
            raise TypeError('axis must be an int')
        res.append([1 if(i == axis) else 0 for i in range(3)])
        if not forward_only:
            res.append([-1 if(i == axis) else 0 for i in range(3)])

    return res


def combine(x_vals, y_vals, z_vals):
    """
    Creates all possible combinations of the given values. ``z`` changes \
    fastest, ``x`` slowest.

    :param x_vals:      Possible values for ``x``
    :type x_vals:       int or list(int)
    :param y_vals:      Possible values for ``y``
    :type y_vals:       int or list(int)
    :param z_vals:      Possible values for ``z``
    :type z_vals:       int or list(int)
    """
    res = []
    try:
        for x in x_vals:
            res.extend(combine(x, y_vals, z_vals))
    except TypeError:
        try:
            for y in y_vals:
                res.extend(combine(x_vals, y, z_vals))
        except TypeError:
            try:
                for z in z_vals:
                    res.extend(combine(x_vals, y_vals, z))
            except TypeError:
                res.append([x_vals, y_vals, z_vals])
    return res

