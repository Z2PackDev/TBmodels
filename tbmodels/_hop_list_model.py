#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    05.10.2015 01:04:24 CEST
# File:    _hoppings_list_model.py

from ._tb_model import Model
import ptools.sparse_matrix as sp

import numpy as np
import collections as co

class HopListModel(Model):
    r"""
    Describes a tight-binding model set up via list of hoppings.

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

    def __init__(self, size, on_site=None, hop_list=[], pos=None, occ=None, add_cc=True, uc=None):
        class _hop:
            """
            POD for hoppings
            """
            def __init__(self):
                self.data = []
                self.row_idx = []
                self.col_idx = []

            def append(self, data, row_idx, col_idx):
                self.data.append(data)
                self.row_idx.append(row_idx)
                self.col_idx.append(col_idx)

        # create data, row_idx, col_idx for setting up the CSR matrices
        hop_list_dict = co.defaultdict(lambda: _hop())
        for i, j, R, t in hop_list:
            R_vec = tuple(R)
            hop_list_dict[R_vec].append(t, i, j)
            if add_cc:
                hop_list_dict[R_vec].append(t.conjugate(), j, i)

        # creating CSR matrices
        hop_dict = dict()
        for key, val in hop_list_dict.items():
            hop_dict[key] = sp.csr((val.data, (val.row_idx, val.col_idx)), dtype=complex)
        super(HopListModel, self).__init__(on_site=on_site, hop=hop_dict, pos=pos, occ=occ, uc=uc)
