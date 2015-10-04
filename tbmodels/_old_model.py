#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    05.10.2015 01:04:24 CEST
# File:    _old_model.py

from ._tb_model import Model

class OldModel(Model):

    def __init__(self):
        r"""
        Describes a tight-binding model.

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

        
        def _create_model(self, in_place, on_site, hop, pos=None, occ=None, add_cc=True, uc=None):
        """
        Creates a new model if in_place=False and modifies the current one else.
        """
        if in_place:
            if uc is None:
                self._uc = None
            else:
                self._uc = np.array(uc)
            self._on_site = np.array(on_site, dtype=float)

            # take pos if given, else default to [0., 0., 0.] * number of orbitals
            if pos is None:
                self.pos = [np.array([0., 0., 0.]) for _ in range(len(self._on_site))]
                uc_offset = [np.array([0, 0, 0], dtype=int) for _ in range(len(self._on_site))]
            # all positions are mapped into the home unit cell
            elif len(pos) == len(self._on_site):
                self.pos = [np.array(p) % 1 for p in pos]
                uc_offset = [np.array(np.floor(p), dtype=int) for p in pos]
            else:
                raise ValueError('invalid argument for "pos": must be either None or of the same length as the number of orbitals (on_site)')

            # adding hoppings and complex conjugates if required
            self._hop = [[i0, i1, np.array(G, dtype=int) + uc_offset[i1] - uc_offset[i0], t] for i0, i1, G, t in hop]
            if add_cc:
                self._hop.extend([[i1, i0, -np.array(G, dtype=int) - uc_offset[i1] + uc_offset[i0], t.conjugate()] for i0, i1, G, t in hop])

            # take occ if given, else default to half the number of orbitals
            if occ is None:
                self.occ = int(len(on_site) / 2)
            else:
                self.occ = int(occ)
            # for the precomputation of Hamilton terms
            self._unchanged = False
            return
        else:
            return Model(on_site, hop, pos, occ, add_cc, uc)


        # PRECOMPUTE FUNCTION
        hamilton_diag = np.array(np.diag(self._on_site), dtype=complex)
        # sort unique G's
        G_key = lambda x: tuple(x[2])
        G_list = list(sorted(list(set([tuple(G_key(x)) for x in self._hop]))))
        hamilton_parts = []
        num_hop_added = 0
        # split hoppings according to their G
        G_splitted_hop = [list(x) for _, x in itertools.groupby(sorted(self._hop, key=G_key), key=G_key)]
        for G_group in G_splitted_hop:
            tmp_hamilton_parts = np.zeros_like(hamilton_diag, dtype=complex)
            for i0, i1, _, t in G_group:
                tmp_hamilton_parts[i0, i1] += t
                num_hop_added += 1
            # each sparse matrix in _hamilton_parts corresponds to a certain G
            hamilton_parts.append(sparse.coo_matrix(tmp_hamilton_parts, dtype=complex))
        assert num_hop_added == len(self._hop)
        self.bare_model = BareModel(
            occ=self.occ,
            hamilton_diag=hamilton_diag,
            G_list=G_list,
            hamilton_parts=hamilton_parts
        )
