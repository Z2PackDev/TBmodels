#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    02.06.2015 17:50:33 CEST
# File:    _tb_model.py

from __future__ import division, print_function

from .ptools.locker import Locker
from .ptools import sparse_matrix as sp

import six
import copy
import time
import warnings
import functools
import numpy as np
import collections as co
import scipy.linalg as la

@six.add_metaclass(Locker)
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
        return la.eigh(self.hamilton(k))[0]

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
        if num_g == 0:
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


    def __str__(self):
        res = self._entries_section('general', dict(occ=self.occ, dim=self.dim, size=self.size)) + '\n'
        res += self._array_section('pos', self.pos) + '\n'
        if self.uc is not None:
            res += self._array_section('uc', self.uc.T) + '\n'
        res += self._hop_section('hop')
        return res

    #--------------------- STR HELPER FUNCTIONS ----------------------------#

    def _entries_section(self, name, entries):
        """
        """
        # Only supports simple data for now. Expand when necessary.
        res = ''
        lines = []
        for key, value in entries.items():
            if value is not None:
                lines.append('{0} = {1}'.format(key, value))
        res += '\n'.join(lines)
        return self._section(name, res)

    def _array_section(self, name, array):
        """
        Writes an array consisting of a 2D floating point array
        """
        res = ''
        max_width = 0
        for line in array:
            for entry in line:
                max_width = max(max_width, len('{: .100g}'.format(entry)))
        for line in array:
            res += (' ' * 4).join(['{: .100g}'.format(entry).ljust(max_width) for entry in line]).rstrip(' ') + '\n'

        return self._section(name, res)

    def _hop_section(self, name):
        """
        Writes the [hop] section
        """
        max_width_idx = len(str(self.size))
        content = ''
        tab_char = ' ' * 4
        format_str = (
            tab_char + '{{0:< {0}}}' + tab_char + '{{1:< {0}}}' + tab_char +
            '{{2: .100g}}'
        ).format(max_width_idx)
        for key in sorted(self.hop.keys()):
            content += '(' + ' '.join(str(x) for x in key) + ')' + '\n'
            mat = self.hop[key]
            for i, j in zip(*mat.nonzero()):
                content += format_str.format(i, j, mat[i, j]) + '\n'
            content += '\n'
        content = content.rstrip('\n')
        return self._section(name, content)


    @staticmethod
    def _section(name, content):
        """Wraps a string into section form, with a given name."""
        return '[{0}]\n'.format(name) + content + '\n'

    #--------------------- END STR HELPER FUNCTIONS ------------------------#

    #-------------------MODIFYING THE MODEL ----------------------------#
    def add_hop(self, overlap, orbital_1, orbital_2, R):
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

    def add_on_site(self, energy, orbital):
        """
        TODO
        """
        self.add_hop(energy / 2., orbital, orbital, self._zero_vec)

    #-------------------CREATING DERIVED MODELS-------------------------#
    #---- arithmetic operations ----#
    def __add__(self, model):
        """
        TODO
        """
        if not isinstance(model, Model):
            raise ValueError('Invalid argument type for Model.__add__: {}'.format(type(model)))

        # ---- CONSISTENCY CHECKS ----
        # check if the occupation number matches
        if self.occ != model.occ:
            raise ValueError('Error when adding Models: occupation numbers ({0}, {1}) don\'t match'.format(self.occ, model.occ))

        # check if the size of the hopping matrices match
        if self.size != model.size:
            raise ValueError('Error when adding Models: the number of states ({0}, {1}) doesn\'t match'.format(len(self.size), len(model.size)))

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

    def __radd__(self, model):
        """
        Addition is commutative.
        """
        return self.__add__(model)

    def __sub__(self, model):
        """
        TODO
        """
        return self + -model

    def __neg__(self):
        """
        TODO
        """
        return -1 * self


    def __mul__(self, x):
        """
        Multiply hopping parameter strengths by a constant factor.
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
        Multiplication with constant factors is commutative.
        """
        return self.__mul__(x)

    def __div__(self, x):
        """
        Division by a constant factor.
        """
        return self * (1. / x)

    # for Python 3
    def __truediv__(self, x):
        """
        Division by a constant factor.
        """
        return self.__div__(x)

    #---- other derived models ----#
    def supercell(self, dim, periodic=(True, True, True), passivation=None):
        r"""
        Creates a tight-binding model which describes a supercell.

        :param dim: The dimensions of the supercell in terms of the previous unit cell.
        :type dim:  list(int)

        :param periodic:    Determines whether periodicity is kept in each crystal direction. If not (entry is ``False``), hopping terms that go across the border of the supercell (in the given direction) are cut.
        :type periodic:     list(bool)

        :param passivation: Determines the passivation on the surface layers. It must be a function taking three input variables ``x, y, z``, which are lists ``[bottom, top]`` of booleans indicating whether a given unit cell inside the supercell touches the bottom and top edge in the given direction. The function returns a list of on-site energies (must be the same length as the initial number of orbitals) determining the passivation strength in said unit cell.
        :type passivation:  function
        """
        dim = np.array(dim, dtype=int)
        nx, ny, nz = dim

        new_occ = None if self.occ is None else sum(dim) * self.occ
        if self.uc is None:
            new_uc = None
        else:
            new_uc = self.uc * dim

        # the new positions, normalized to the supercell
        new_pos = []
        reduced_pos = [p / dim for p in self.pos]
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    tmp_offset = np.array([i, j, k]) / dim
                    for p in reduced_pos:
                        new_pos.append(tmp_offset + p)

        # new hoppings, cutting those that cross the supercell boundary
        # in a non-periodic direction
        new_size = self.size * nx * ny * nz
        new_hop = co.defaultdict(lambda: np.zeros((new_size, new_size), dtype=complex))
        # full index of an orbital in unit cell at uc_pos
        def full_idx(uc_pos, orbital_idx):
            """
            Computes the full index of an orbital in a given unit cell.
            """
            uc_idx = _pos_to_idx(uc_pos, dim)
            return uc_idx * self.size + orbital_idx

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    uc0_pos = np.array([i, j, k], dtype=int)
                    for R, hop_mat in self.hop.items():
                        hop_mat = np.array(hop_mat)
                        for i0, row in enumerate(hop_mat):
                            for i1, t in enumerate(row):
                                # new index of orbital 0
                                new_i0 = full_idx(uc0_pos, i0)
                                # position of the uc of orbital 1, not mapped inside supercell
                                full_uc1_pos = uc0_pos + np.array(R)
                                outside_supercell = [(p < 0) or (p >= d) for p, d in zip(full_uc1_pos, dim)]
                                # test if the hopping should be cut
                                cut_hop = any([not per and outside for per, outside in zip(periodic, outside_supercell)])
                                if cut_hop:
                                    continue
                                else:
                                    # R in terms of supercells
                                    new_R = np.array(np.floor(full_uc1_pos / dim), dtype=int)
                                    # mapped into the supercell
                                    uc1_pos = full_uc1_pos % dim
                                    new_i1 = full_idx(uc1_pos, i1)
                                    new_hop[tuple(new_R)][new_i0, new_i1] += t

        # new on_site terms, including passivation
        if passivation is None:
            passivation = lambda x, y, z: np.zeros(self.size)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    idx = (i * ny * nz + j * nz + k) * self.size
                    new_hop[self._zero_vec][idx:idx + self.size, idx:idx + self.size] += np.diag(np.array(passivation(*_edge_detect_pos([i, j, k], dim)), dtype=float) * 0.5)
        return Model(
            hop=new_hop,
            pos=new_pos,
            occ=new_occ,
            uc=new_uc,
            contains_cc=False
        )

    def trs(self):
        """
        Adds a time-reversal image of the current model.
        """
        # doubling the occupation number and positions
        new_occ = None if (self.occ is None) else self.occ * 2
        new_pos = np.vstack([self.pos, self.pos])
        new_hop = dict()
        # doubling the hopping terms
        for R, hop in self.hop.items():
            if R not in new_hop.keys():
                new_hop[R] = np.zeros((2 * self.size, 2 * self.size), dtype=complex)
            new_hop[R][:self.size, :self.size] += hop
            # here you can either do -R  or hop.conjugate() or hop.T, but not both
            new_hop[R][self.size:, self.size:] += hop.conjugate()
        return Model(
            hop=new_hop,
            occ=new_occ,
            pos=new_pos,
            uc=self.uc,
            contains_cc=False
        )

    def change_uc(self, uc):
        """
        Creates a new model with a different unit cell. The new unit cell must have the same volume as the previous one, i.e. the number of atoms per unit cell stays the same, and cannot change chirality.

        :param uc: The new unit cell, given w.r.t. to the old one. Lattice vectors are given as column vectors in a 3x3 matrix.
        """
        uc = np.array(uc)
        if la.det(uc) != 1:
            raise ValueError('The determinant of uc is {0}, but should be 1'.format(la.det(uc)))
        if self.uc is not None:
            new_uc = np.dot(self.uc, uc)
        else:
            new_uc = None
        new_pos = [la.solve(uc, p) for p in self.pos]
        new_hop = {tuple(np.array(la.solve(uc, R), dtype=int)): hop_mat for R, hop_mat in self.hop.items()}

        return Model(
            hop=new_hop,
            pos=new_pos,
            occ=self.occ,
            uc=new_uc,
            contains_cc=False,
        )

    def em_field(self, scalar_pot=None, vec_pot=None, prefactor_scalar=1, prefactor_vec=7.596337572e-6, mode_scalar='relative', mode_vec='relative'):
        r"""
        Creates a model including an electromagnetic field described by a scalar potential :math:`\Phi(\mathbf{r})` and a vector potential :math:`\mathbf{A}(\mathbf{r})` .

        :param scalar_pot:  A function returning the scalar potential given the position as a numpy ``array`` of length 3.
        :type scalar_pot:   function

        :param vec_pot: A function returning the vector potential (``list`` or ``numpy array`` of length 3) given the position as a numpy ``array`` of length 3.
        :type vec_pot:  function

        The units in which the two potentials are given can be determined by specifying a multiplicative prefactor. By default, the scalar potential is given in :math:`\frac{\text{energy}}{\text{electron}}` in the given energy units, and the scalar potential is given in :math:`\text{T} \cdot {\buildrel _{\circ} \over {\mathrm{A}}}`, assuming that the unit cell is also given in Angstrom.

        Given a ``prefactor_scalar`` :math:`p_s` and ``prefactor_vec`` :math:`p_v`, the on-site energies are modified by

        :math:`\epsilon_{\alpha, \mathbf{R}} = \epsilon_{\alpha, \mathbf{R}}^0 + p_s \Phi(\mathbf{R})`

        and the hopping terms are transformed by

        :math:`t_{\alpha^\prime , \alpha } (\mathbf{R}, \mathbf{R}^\prime) = t_{\alpha^\prime , \alpha }^0 (\mathbf{R}, \mathbf{R}^\prime) \times \exp{\left[ -i ~ p_v~(\mathbf{R}^\prime - \mathbf{R})\cdot(\mathbf{A}(\mathbf{R}^\prime ) - \mathbf{A}(\mathbf{R})) \right]}`

        :param prefactor_scalar:    Prefactor determining the unit of the scalar potential.
        :type prefactor_scalar:     float

        :param prefactor_vec:       Prefactor determining the unit of the vector potential.
        :type prefactor_vec:        float

        The positions :math:`\mathbf{r}` given to the potentials :math:`\Phi` and :math:`\mathbf{A}` can be either absolute or relative to the unit cell:

        :param mode_scalar: Determines whether the input for the ``scalar_pot`` function is given as an absolute position (``mode_scalar=='absolute'``) or relative to the unit cell (``mode_scalar=='relative'``).
        :type mode_scalar:  str

        :param mode_vec:    Determines whether the input for the ``vec_pot`` function is given as an absolute position (``mode_vec=='absolute'``) or relative to the unit cell (``mode_vec=='relative'``).
        :type mode_vec:     str
        """
        new_hop = copy.deepcopy(self.hop)
        if scalar_pot is not None:
            for i, p in enumerate(self.pos):
                if mode_scalar == 'relative':
                    new_hop[self._zero_vec][i, i] += 0.5 * prefactor_scalar * scalar_pot(p)
                elif mode_scalar == 'absolute':
                    new_hop[self._zero_vec][i, i] += 0.5 * prefactor_scalar * scalar_pot(np.dot(self.uc, p))
                else:
                    raise ValueError('Unrecognized value for mode_scalar. Must be either "absolute" or "relative"')

        if vec_pot is not None:
            warnings.warn('The code for non-zero vector potential has not been tested at all!', UserWarning)
            vector_pot = lambda r: np.array(vec_pot(r))
            if self.uc is None:
                raise ValueError('Unit cell is not specified')
            for R, hop_mat in self.hop.items():
                for i0, i1 in np.vstack(hop_mat.nonzero()).T:
                    p0 = self.pos[i0]
                    p1 = self.pos[i0]
                    r0 = np.dot(self.uc, p0)
                    r1 = np.dot(self.uc, p1)
                    if mode_vec == 'absolute':
                        # project into the home UC
                        A0 = vector_pot(np.dot(self.uc, p0 % 1))
                        A1 = vector_pot(np.dot(self.uc, p1 % 1))
                    elif mode_vec == 'relative':
                        # project into the home UC
                        A0 = vector_pot(p0 % 1)
                        A1 = vector_pot(p1 % 1)
                    else:
                        raise ValueError('Unrecognized value for mode_vec. Must be either "absolute" or "relative"')
                    hop_mat[i0, i1] *= np.exp(-1j * prefactor_vec * np.dot(R + r1 - r0, A1 - A0))

        return Model(
            hop=new_hop,
            pos=self.pos,
            occ=self.occ,
            uc=self.uc,
            contains_cc=False,
        )

#----------------HELPER FUNCTIONS FOR SUPERCELL-------------------------#
def _pos_to_idx(pos, dim):
    """index -> position"""
    for p, d in zip(pos, dim):
        if p >= d:
            raise IndexError('pos is out of bounds')
    return ((pos[0] * dim[1]) + pos[1]) * dim[2] + pos[2]

def _edge_detect_pos(pos, dim):
    """detect edges of the supercell"""
    for p, d in zip(pos, dim):
        if p >= d:
            raise IndexError('pos is out of bounds')
    edges = [[None] * 2 for i in range(3)]
    for i in range(3):
        edges[i][0] = (pos[i] == 0)
        edges[i][1] = (pos[i] == dim[i] - 1)
    return edges
