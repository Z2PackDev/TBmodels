#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    05.10.2015 01:04:24 CEST
# File:    _hoppings_list_model.py

from ._tb_model import Model

import numpy as np

class HoppingsListModel(Model):
    r"""
    Describes a tight-binding model set up via list of hoppings.

    :param on_site: On-site energies of the orbitals.
    :type on_site:  list

    :param hop: Hopping terms. Each hopping terms is a list
        [O1, O2, G, t] where O1 and O2 are the indices of the two orbitals
        involved, G is the reciprocal lattice vector indicating the UC
        where O2 is located (if O1 is located in the home UC), and t
        is the hopping strength.
    :type hop: list

    :param pos:   Positions of the orbitals. By default (positions = ``None``),
        all orbitals are put at the origin.
    :type pos:    list

    :param occ: Number of occupied states. Default: Half the number of orbitals.
    :type occ:  int

    :param add_cc:  Determines whether the complex conjugates of the hopping
        parameters are added automatically. Default: ``True``.
    :type add_cc:   bool

    :param uc: Unit cell of the system. The lattice vectors :math:`a_i` are to be given as column vectors. By default, no unit cell is specified, meaning an Error will occur when adding electromagnetic field.
    :type uc: 3x3 matrix
    """

    def __init__(self, size, hoppings_list, pos=None, occ=None, add_cc=True, uc=None):
        hoppings_dict = dict()
        #~ hoppings[(0, 0, 0)] = np.array(np.diag(on_site), dtype=complex)
        for i, j, G, t in hoppings_list:
            G_vec = tuple(G)
            if G_vec not in hoppings_dict.keys():
                hoppings_dict[G_vec] = np.zeros((size, size), dtype=complex)
            hoppings_dict[G_vec][i, j] += t
            if add_cc:
                hoppings_dict[G_vec][j, i] += t.conjugate()
        super(HoppingsListModel, self).__init__(hoppings_dict, pos=pos, occ=occ, uc=uc)
