#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    02.06.2015 17:50:33 CEST
# File:    _tb_model.py

from __future__ import division, print_function

from ._ptools import sparse_matrix as sp

import six
import copy
import time
import warnings
import functools
import numpy as np
import collections as co
import scipy.linalg as la

class Model(object):
    r"""

    :param hop:    Hopping matrices, as a dict containing the corresponding R as a key.
    :type hop:     dict

    :param size:        Number of states. Defaults to the size of the hopping matrices, if those are given.
    :type size:         int

    :param occ:         Number of occupied states.
    :type occ:          int

    :param pos:         Positions of the atoms. Defaults to [0., 0., 0.]. Must be in the home UC.
    :type pos:          list(array)

    :param contains_cc: Whether the full overlaps are given, or only the reduced representation which does not contain the complex conjugate terms (and only half the zero-terms).
    :type contains_cc:  bool

    :param cc_tol:    Tolerance when the complex conjugate terms are checked for consistency.
    :type cc_tol:     float
    """
    def __init__(self, on_site=None, hop=None, size=None, dim=None, occ=None, pos=None, uc=None, contains_cc=True, cc_tol=1e-12):
        if hop is None:
            hop = dict()
        # ---- SIZE ----
        self._init_size(size=size, on_site=on_site, hop=hop)

        # ---- DIMENSION ----
        self._init_dim(dim=dim, hop=hop, pos=pos)

        # ---- HOPPING TERMS AND POSITIONS ----
        self._init_hop_pos(
            on_site=on_site,
            hop=hop,
            pos=pos,
            contains_cc=contains_cc,
            cc_tol=cc_tol
        )

        # ---- CONSISTENCY CHECK FOR SIZE ----
        self._check_size_hop()
        # ---- CONSISTENCY CHECK FOR DIM ----

        # ---- UNIT CELL ----
        self.uc = None if uc is None else np.array(uc) # implicit copy

        # ---- OCCUPATION NR ----
        self.occ = None if (occ is None) else int(occ)

    #---------------- INIT HELPER FUNCTIONS --------------------------------#
    def _init_size(self, size, on_site, hop):
        """
        Sets the size of the system (number of orbitals).
        """
        if size is not None:
            self.size = size
        elif on_site is not None:
            self.size = len(on_site)
        elif len(hop) != 0:
            self.size = six.next(six.itervalues(hop)).shape[0]
        else:
            raise ValueError('Empty hoppings dictionary supplied and no size given. Cannot determine the size of the system.')

    def _init_dim(self, dim, hop, pos):
        r"""
        Sets the system's dimensionality.
        """
        if dim is not None:
            self.dim = dim
        elif pos is not None:
            self.dim = len(pos[0])
        elif len(hop.keys()) > 0:
            self.dim = len(next(iter(hop.keys())))
        else:
            raise ValueError('No dimension specified and no positions or hoppings are given. The dimensionality of the system cannot be determined.')

        self._zero_vec = tuple([0] * self.dim)

    def _init_hop_pos(self, on_site, hop, pos, contains_cc, cc_tol):
        """
        Sets the hopping terms and positions, mapping the positions to the UC (and changing the hoppings accordingly) if necessary.
        """
        hop = {tuple(key): sp.csr(value, dtype=complex) for key, value in hop.items()}
        # positions
        if pos is None:
            self.pos = [np.array(self._zero_vec) for _ in range(self.size)]
        elif len(pos) == self.size:
            pos, hop = self._map_to_uc(pos, hop)
            self.pos = np.array(pos) # implicit copy
        else:
            raise ValueError('invalid argument for "pos": must be either None or of the same length as the size of the system')
        if contains_cc:
            hop = self._reduce_hop(hop, cc_tol)
        else:
            hop = self._map_hop_positive_R(hop)
        # use partial instead of lambda to allow for pickling
        self.hop = co.defaultdict(
            functools.partial(sp.csr, (self.size, self.size), dtype=complex)
        )
        for R, h_mat in hop.items():
            self.hop[R] = sp.csr(h_mat)
        # add on-site terms
        if on_site is not None:
            if len(on_site) != self.size:
                raise ValueError('The number of on-site energies {0} does not match the size of the system {1}'.format(len(on_site), self.size))
            self.hop[self._zero_vec] += 0.5 * sp.csr(np.diag(on_site))

    # helpers for _init_hop_pos
    def _map_to_uc(self, pos, hop):
        """
        hoppings in csr format
        """
        uc_offsets = [np.array(np.floor(p), dtype=int) for p in pos]
        # ---- common case: already mapped into the UC ----
        if all([all(o == 0 for o in offset) for offset in uc_offsets]):
            return pos, hop

        # ---- uncommon case: handle mapping ----
        new_pos = [np.array(p) % 1 for p in pos]
        new_hop = co.defaultdict(lambda: np.zeros((self.size, self.size), dtype=complex))
        for R, hop_mat in hop.items():
            hop_mat = np.array(hop_mat)
            for i0, row in enumerate(hop_mat):
                for i1, t in enumerate(row):
                    if t != 0:
                        R_new = tuple(np.array(R, dtype=int) + uc_offsets[i1] - uc_offsets[i0])
                        new_hop[R_new][i0][i1] += t
        new_hop = {key: sp.csr(value) for key, value in new_hop.items()}
        return new_pos, new_hop

    @staticmethod
    def _reduce_hop(hop, cc_tol):
        """
        Reduce the full hoppings representation (with cc) to the reduced one (without cc, zero-terms halved).

        hop is in CSR format
        """
        # Consistency checks
        for R, hop_csr in hop.items():
            if la.norm(hop_csr - hop[tuple(-x for x in R)].T.conjugate()) > cc_tol:
                raise ValueError('The provided hoppings do not correspond to a hermitian Hamiltonian. hoppings[-R] = hoppings[R].H is not fulfilled.')

        res = dict()
        for R, hop_csr in hop.items():
            try:
                if R[np.nonzero(R)[0][0]] > 0:
                    res[R] = hop_csr
                else:
                    continue
            # zero case
            except IndexError:
                res[R] = 0.5 * hop_csr
        return res

    def _map_hop_positive_R(self, hop):
        """
        Maps hoppings with a negative first non-zero index in R to their positive counterpart.
        """
        new_hop = co.defaultdict(lambda: sp.csr((self.size, self.size), dtype=complex))
        for R, hop_csr in hop.items():
            try:
                if R[np.nonzero(R)[0][0]] > 0:
                    new_hop[R] += hop_csr
                else:
                    minus_R = tuple(-x for x in R)
                    new_hop[minus_R] += hop_csr.transpose().conjugate()
            except IndexError:
                new_hop[R] += hop_csr
        return new_hop
    # end helpers for _init_hop_pos

    def _check_size_hop(self):
        """
        Consistency check for the size of the hopping matrices.
        """
        for h_mat in self.hop.values():
            if not h_mat.shape == (self.size, self.size):
                raise ValueError('Hopping matrix of shape {0} found, should be ({1},{1}).'.format(h_mat.shape, self.size))

    def _check_dim(self):
        """Consistency check for the dimension."""
        for key in self.hop.keys():
            if len(key) != self.dim:
                raise ValueError('The length of R = {0} does not match the dimensionality of the system ({1})'.format(key, self.dim))
        for p in self.pos:
            if len(p) != self.dim:
                raise ValueError('The length of position r = {0} does not match the dimensionality of the system ({1})'.format(len(p), self.dim))
        if self.uc is not None:
            if self.uc.shape != (self.dim, self.dim):
                raise ValueError('Inconsistend dimension of the unit cell: {0}, does not match the dimensionality of the system ({1})'.format(self.uc.shape, self.dim))

    #----------------ALTERNATE CONSTRUCTORS---------------------------------#

    @classmethod
    def from_hopping_list(cls, *, size=None, hopping_list=(), **kwargs):
        if size is None:
            try:
                size = len(kwargs['on_site'])
            except KeyError:
                raise ValueError('No on-site energies and no size given. The size of the system cannot be determined.')
        
        class _hop(object):
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
        hop_list_dict = co.defaultdict(_hop)
        for t, i, j, R in hopping_list:
            R_vec = tuple(R)
            hop_list_dict[R_vec].append(t, i, j)

        # creating CSR matrices
        hop_dict = dict()
        for key, val in hop_list_dict.items():
            hop_dict[key] = sp.csr((val.data, (val.row_idx, val.col_idx)), dtype=complex, shape=(size, size))

        return cls(size=size, hop=hop_dict, **kwargs)
        
    @classmethod
    def from_hr(cls, hr_file, *, h_cutoff=None, **kwargs):
        with open(hr_file, 'r') as file_handle:
            num_wann, h_entries = cls._read_hr(file_handle)
            if h_cutoff is None:
                h_cutoff = 0
            
            h_entries = (hopping for hopping in h_entries if abs(hopping[0]) > h_cutoff)

            return cls.from_hopping_list(size=num_wann, hopping_list=h_entries, **kwargs)

    @staticmethod
    def _read_hr(file_handle):
        r"""
        read the number of wannier functions and the hopping entries
        from *hr.dat and converts them into the right format
        """
        file_handle = enumerate(file_handle)
        next(file_handle) # skip first line
        num_wann = int(next(file_handle)[1])
        nrpts = int(next(file_handle)[1])

        # get degeneracy points
        deg_pts = []
        # order in zip important because else the next data element is consumed
        for _, (_, line) in zip(range(int(np.ceil(nrpts / 15))), file_handle):
            deg_pts.extend(int(x) for x in line.split())
        assert len(deg_pts) == nrpts

        num_wann_square = num_wann**2
        def to_entry(line, i):
            """Turns a line (string) into a hop_list entry"""
            line_no, line = line
            entry = line.split()
            orbital_a = int(entry[3]) - 1
            orbital_b = int(entry[4]) - 1
            # test consistency of orbital numbers
            if not (
                orbital_a == i % num_wann and 
                orbital_b == (i % num_wann_square) // num_wann
            ):
                raise ValueError(
                    'Inconsistent orbital numbers in line number {}'.format(line_no + 1)
                )
            return [
                (float(entry[5]) + 1j * float(entry[6])) / (deg_pts[i // num_wann_square]),
                orbital_a,
                orbital_b,
                [int(x) for x in entry[:3]]
            ]

        # skip random empty lines
        lines_nonempty = (l for l in file_handle if l[1].strip())
        hop_list = (to_entry(line, i) for i, line in enumerate(lines_nonempty))

        return num_wann, hop_list

    #---------------- BASIC FUNCTIONALITY ----------------------------------#
    def hamilton(self, k):
        """
        Creates the Hamiltonian matrix.

        :param k:   k-point
        :type k:    list

        :returns:   2D numpy array
        """
        k = np.array(k)
        H = sum(np.array(hop) * np.exp(2j * np.pi * np.dot(R, k)) for R, hop in self.hop.items())
        H += H.conjugate().T
        return np.array(H)

    def eigenval(self, k):
        """
        Returns the eigenvalues at a given k point.

        :param k:   k-point
        :type k:    list

        :returns:   list of eigenvalues
        """
        return la.eigvalsh(self.hamilton(k))

    def to_hr(self):
        """
        Returns a string containing the model in Wannier90's ``*_hr.dat`` format.

        :returns: str
        """
        lines = []
        tagline = ' created by the TBModels package    ' + time.strftime('%a, %d %b %Y %H:%M:%S %Z')
        lines.append(tagline)
        lines.append('{0:>12}'.format(self.size))
        num_g = len(self.hop.keys()) * 2 - 1
        if num_g <= 0:
            raise ValueError('Cannot print empty model to hr format.')
        lines.append('{0:>12}'.format(num_g))
        tmp = ''
        for i in range(num_g):
            if tmp != '' and i % 15 == 0:
                lines.append(tmp)
                tmp = ''
            tmp += '    1'
        lines.append(tmp)

        # negative
        for R in reversed(sorted(self.hop.keys())):
            if R != self._zero_vec:
                minus_R = tuple(-x for x in R)
                lines.extend(self._mat_to_hr(
                    minus_R, self.hop[R].conjugate().transpose()
                ))
        # zero
        if self._zero_vec in self.hop.keys():
            lines.extend(self._mat_to_hr(
                self._zero_vec,
                self.hop[self._zero_vec] + self.hop[self._zero_vec].conjugate().transpose()
            ))
        # positive
        for R in sorted(self.hop.keys()):
            if R != self._zero_vec:
                lines.extend(self._mat_to_hr(
                    R, self.hop[R]
                ))

        return '\n'.join(lines)

    @staticmethod
    def _mat_to_hr(R, mat):
        """
        Creates the ``*_hr.dat`` string for a single hopping matrix.
        """
        lines = []
        mat = np.array(mat).T # to be consistent with W90's ordering
        for j, column in enumerate(mat):
            for i, t in enumerate(column):
                lines.append(
                    '{0[0]:>5}{0[1]:>5}{0[2]:>5}{1:>5}{2:>5}{3.real:>12.6f}{3.imag:>12.6f}'.format(R, i + 1, j + 1, t)
                )
        return lines

    def __repr__(self):
        return ' '.join('tbmodels.Model(hop={1}, pos={0.pos!r}, uc={0.uc!r}, occ={0.occ}, contains_cc=False)'.format(self, dict(self.hop)).replace('\n', ' ').replace('array', 'np.array').split())

    #-------------------MODIFYING THE MODEL ----------------------------#
    def add_hopping(self, overlap, orbital_1, orbital_2, R):
        r"""
        Adds a hopping term of a given overlap between an orbital in the home unit cell (``orbital_1``) and another orbital (``orbital_2``) located in the unit cell pointed to by ``R``.

        The complex conjugate of the hopping is added automatically. That is, the hopping from ``orbital_2`` to ``orbital_1`` with conjugated ``overlap`` and inverse ``R`` does not have to be added manually.

        .. note::
            This means that adding a hopping of overlap :math:`\epsilon` between an orbital and itself in the home unit cell increases the orbitals on-site energy by :math:`2 \epsilon`.

        :param overlap:    Strength of the hopping term (in energy units).
        :type overlap:     complex

        :param orbital_1:   Index of the first orbital.
        :type orbital_1:    int

        :param orbital_2:   Index of the second orbital.
        :type orbital_2:    int

        :param R:           Lattice vector pointing to the unit cell where `orbital_2` lies.
        :type R:            list(int)

        .. warning::
            The positions given in the constructor of :class:`Model` are automatically mapped into the home unit cell. This has to be taken into account when determining ``R``.

        """
        R = tuple(R)
        mat = np.zeros((self.size, self.size), dtype=complex)
        try:
            if R[np.nonzero(R)[0][0]] > 0:
                mat[orbital_1, orbital_2] = overlap
            else:
                R = tuple(-x for x in R)
                mat[orbital_2, orbital_1] = overlap.conjugate()
        except IndexError:
            mat[orbital_1, orbital_2] = overlap
        self.hop[R] += sp.csr(mat)

    def add_on_site(self, on_site):
        """
        TODO
        """
        if self.size != len(on_site):
            raise ValueError('The number of on-site energy terms should be {}, but is {}.'.format(self.size, len(on_site)))
        for orbital, energy in enumerate(on_site):
            self.add_hopping(energy / 2., orbital, orbital, self._zero_vec)

    #-------------------CREATING DERIVED MODELS-------------------------#
    #---- arithmetic operations ----#
    def __add__(self, model):
        """
        Adds two models together by adding their hopping terms.
        """
        if not isinstance(model, Model):
            raise ValueError('Invalid argument type for Model.__add__: {}'.format(type(model)))

        # ---- CONSISTENCY CHECKS ----
        # check if the occupation number matches
        if self.occ != model.occ:
            raise ValueError('Error when adding Models: occupation numbers ({0}, {1}) don\'t match'.format(self.occ, model.occ))

        # check if the size of the hopping matrices match
        if self.size != model.size:
            raise ValueError('Error when adding Models: the number of states ({0}, {1}) doesn\'t match'.format(self.size, model.size))

        # check if the unit cells match
        uc_match = True
        if self.uc is None or model.uc is None:
            if model.uc is not self.uc:
                uc_match = False
        else:
            tolerance = 1e-6
            for v1, v2 in zip(self.uc, model.uc):
                if not uc_match:
                    break
                for x1, x2 in zip(v1, v2):
                    if abs(x1 - x2) > tolerance:
                        uc_match = False
                        break
        if not uc_match:
            raise ValueError('Error when adding Models: unit cells don\'t match.\nModel 1: {0.pos}\nModel 2: {1.pos}'.format(self, model))

        # check if the positions match
        pos_match = True
        tolerance = 1e-6
        for v1, v2 in zip(self.pos, model.pos):
            if not pos_match:
                break
            for x1, x2 in zip(v1, v2):
                if abs(x1 - x2) > tolerance:
                    pos_match = False
                    break
        if not pos_match:
            raise ValueError('Error when adding Models: positions don\'t match.\nModel 1: {0.pos}\nModel 2: {1.pos}'.format(self, model))

        # ---- MAIN PART ----
        new_hop = copy.deepcopy(self.hop)
        for R, hop_mat in model.hop.items():
            new_hop[R] += hop_mat
        # -------------------
        return Model(
            hop=new_hop,
            pos=self.pos,
            occ=self.occ,
            uc=self.uc,
            contains_cc=False,
        )

    def __sub__(self, model):
        """
        Substracts one model from another by substracting all hopping terms.
        """
        return self + -model

    def __neg__(self):
        """
        Changes the sign of all hopping terms.
        """
        return -1 * self


    def __mul__(self, x):
        """
        Multiplies hopping terms by x.
        """
        new_hop = dict()
        for R, hop_mat in self.hop.items():
            new_hop[R] = x * hop_mat

        return Model(
            hop=new_hop,
            pos=self.pos,
            occ=self.occ,
            uc=self.uc,
            contains_cc=False,
        )

    def __rmul__(self, x):
        """
        Multiplies hopping terms by x.
        """
        return self.__mul__(x)

    def __div__(self, x):
        """
        Divides hopping terms by x.
        """
        return self * (1. / x)

    # for Python 3
    def __truediv__(self, x):
        """
        Divides hopping terms by x.
        """
        return self.__div__(x)
