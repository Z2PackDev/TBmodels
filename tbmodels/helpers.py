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

def matrix_to_hoppings(mat, orbitals=None, G=(0, 0, 0), multiplier=1.):
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
