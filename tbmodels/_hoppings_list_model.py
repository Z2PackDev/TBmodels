#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    05.10.2015 01:04:24 CEST
# File:    _hoppings_list_model.py

from ._tb_model import Model

import numpy as np

class HoppingsListModel(Model):

    def __init__(self, size, hoppings_list, pos=None, occ=None, add_cc=True, uc=None):
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
        hoppings_dict = dict()
        #~ hoppings[(0, 0, 0)] = np.array(np.diag(on_site), dtype=complex)
        for i0, i1, G, t in hoppings_list:
            G_vec = tuple(G)
            if G_vec not in hoppings_dict.keys():
                hoppings_dict[G_vec] = np.zeros((size, size), dtype=complex)
            hoppings_dict[G_vec][i0, i1] += t
            if add_cc:
                hoppings_dict[G_vec][i1, i0] += t.conjugate()
        super(HoppingsListModel, self).__init__(hoppings_dict, pos=pos, occ=occ, uc=uc)
        
        
        #~ # PRECOMPUTE FUNCTION
        #~ hamilton_diag = np.array(np.diag(self._on_site), dtype=complex)
        #~ # sort unique G's
        #~ G_key = lambda x: tuple(x[2])
        #~ G_list = list(sorted(list(set([tuple(G_key(x)) for x in self._hop]))))
        #~ hamilton_parts = []
        #~ num_hop_added = 0
        #~ # split hoppings according to their G
        #~ G_splitted_hop = [list(x) for _, x in itertools.groupby(sorted(self._hop, key=G_key), key=G_key)]
        #~ for G_group in G_splitted_hop:
            #~ tmp_hamilton_parts = np.zeros_like(hamilton_diag, dtype=complex)
            #~ for i0, i1, _, t in G_group:
                #~ tmp_hamilton_parts[i0, i1] += t
                #~ num_hop_added += 1
            #~ # each sparse matrix in _hamilton_parts corresponds to a certain G
            #~ hamilton_parts.append(sparse.coo_matrix(tmp_hamilton_parts, dtype=complex))
        #~ assert num_hop_added == len(self._hop)
        #~ self.bare_model = BareModel(
            #~ occ=self.occ,
            #~ hamilton_diag=hamilton_diag,
            #~ G_list=G_list,
            #~ hamilton_parts=hamilton_parts
        #~ )
