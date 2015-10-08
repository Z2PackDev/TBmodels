#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    02.06.2015 17:50:33 CEST
# File:    _tb_model.py

from __future__ import division, print_function

from mtools.bands import EigenVal
from .ptools import sparse_matrix as sp

import os
import six
import copy
import time
import warnings
import numpy as np
import collections as co
import scipy.linalg as la

class Model(object):
    r"""

    :param hop:    Hopping matrices, as a dict containing the corresponding G as a key.
    :type hop:     dict

    :param size:        Number of states. Defaults to the size of the hopping matrices, if those are given.
    :type size:         int

    :param occ:         Number of occupied states.
    :type occ:          int

    :param pos:         Positions of the atoms. Defaults to [0., 0., 0.]. Must be in the home UC.
    :type pos:          list(array)

    :param contains_cc: Whether the full overlaps are given, or only the reduced representation which does not contain the complex conjugate terms (and only half the zero-terms).
    :type contains_cc:  bool

    :param cc_tolerance:    Tolerance when the complex conjugate terms are checked for consistency.
    :type cc_tolerance:     float
    """
    def __init__(self, on_site=None, hop=dict(), size=None, occ=None, pos=None, uc=None, contains_cc=True, cc_tolerance=1e-12):
        # ---- SIZE ----
        if len(hop) == 0 and size is None and on_site is None:
            raise ValueError('Empty hoppings dictionary supplied and no size given. Cannot determine the size of the system.')
        if size is not None:
            self.size = size
        elif on_site is not None:
            self.size = len(on_site)
        elif len(hop) != 0:
            self.size = six.next(six.itervalues(hop)).shape[0]
        else:
            raise ValueError('Empty hoppings dictionary supplied and no size given. Cannot determine the size of the system.')

        # ---- HOPPING TERMS AND POSITIONS ----
        hop = {tuple(key): sp.csr(value, dtype=complex) for key, value in hop.items()}
        # positions
        if pos is None:
            self.pos = [np.array([0., 0., 0.]) for _ in range(self.size)]
        elif len(pos) == self.size:
            pos, hop = self._map_to_uc(pos, hop)
            self.pos = np.array(pos) # implicit copy
        else:
            raise ValueError('invalid argument for "pos": must be either None or of the same length as the number of orbitals (on_site)')
        if contains_cc:
            hop = self._reduce_hop(hop, cc_tolerance)
        else:
            hop = self._map_hop_positive_G(hop)
        self.hop = co.defaultdict(lambda: sp.csr((self.size, self.size), dtype=complex))
        for G, h_mat in hop.items():
            self.hop[G] = sp.csr(h_mat)
        # add on-site terms
        if on_site is not None:
            if len(on_site) != self.size:
                raise ValueError('The number of on-site energies {0} does not match the size of the system {1}'.format(len(on_site), self.size))
            self.hop[(0, 0, 0)] += sp.csr(np.diag(0.5 * np.array(on_site)))
        
        # consistency check for size
        for h_mat in self.hop.values():
            if not h_mat.shape == (self.size, self.size):
                raise ValueError('Hopping matrix of shape {0} found, should be ({1},{1}).'.format(h_mat.shape, self.size))


        # ---- UNIT CELL ----
        if uc is None:
            self.uc = None
        else:
            self.uc = np.array(uc) # implicit copy

        # ---- OCCUPATION NR ----
        self.occ = None if (occ is None) else int(occ)

    #---------------- INIT HELPER FUNCTIONS --------------------------------#

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
        for G, hop_mat in hop.items():
            hop_mat = np.array(hop_mat)
            for i0, row in enumerate(hop_mat):
                for i1, t in enumerate(row):
                    if t != 0:
                        G_new = tuple(np.array(G, dtype=int) + uc_offsets[i1] - uc_offsets[i0])
                        new_hop[G_new][i0][i1] += t
        new_hop = {key: sp.csr(value) for key, value in new_hop.items()}
        return new_pos, new_hop

    def _reduce_hop(self, hop, cc_tolerance):
        """
        Reduce the full hoppings representation (with cc) to the reduced one (without cc, zero-terms halved).

        hop is in CSR format
        """
        # Consistency checks
        for G, hop_csr in hop.items():
            if la.norm(hop_csr - hop[tuple(-x for x in G)].T.conjugate()) > cc_tolerance:
                raise ValueError('The provided hoppings do not correspond to a hermitian Hamiltonian. hoppings[-G] = hoppings[G].H is not fulfilled.')

        res = dict()
        for G, hop_csr in hop.items():
            if G == (0, 0, 0):
                res[G] = 0.5 * hop_csr
            elif G[np.nonzero(G)[0][0]] > 0:
                res[G] = hop_csr
            else:
                continue
        return res

    def _map_hop_positive_G(self, hop):
        """
        Maps hoppings with a negative first non-zero index in G to their positive counterpart.
        """
        new_hop = co.defaultdict(lambda: sp.csr((self.size, self.size), dtype=complex))
        #~ new_hop = dict()
        for G, hop_csr in hop.items():
            if G == (0, 0, 0):
                new_hop[G] = hop_csr
            elif G[np.nonzero(G)[0][0]] > 0:
                #~ assert(G not in new_hop.keys())
                new_hop[G] += hop_csr
            else:
                minus_G = tuple(-x for x in G)
                #~ assert(minus_G not in new_hop.keys())
                new_hop[minus_G] += hop_csr.transpose().conjugate()
        return new_hop

    #---------------- BASIC FUNCTIONALITY ----------------------------------#
    def hamilton(self, k):
        """
        Creates the Hamiltonian matrix.

        :param k:   k-point
        :type k:    list

        :returns:   2D numpy array
        """
        k = np.array(k)
        H = sum(np.array(hop) * np.exp(2j * np.pi * np.dot(G, k)) for G, hop in self.hop.items())
        H += H.conjugate().T
        return np.array(H)

    def eigenval(self, k):
        """
        Returns the eigenvalues at a given k point.

        :param k:   k-point
        :type k:    list

        :returns:   list of EigenVal objects
        """
        return EigenVal(la.eigh(self.hamilton(k))[0], self.occ)


    def to_hr(self):
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
        for G in reversed(sorted(self.hop.keys())):
            if G != (0, 0, 0):
                minus_G = tuple(-x for x in G)
                lines.extend(self._mat_to_hr(
                    minus_G, self.hop[G].conjugate().transpose()
                ))
        # zero
        zero_G = (0, 0, 0)
        if zero_G in self.hop.keys():
            lines.extend(self._mat_to_hr(
                zero_G,
                self.hop[zero_G] + self.hop[zero_G].conjugate().transpose()
            ))
        # positive
        for G in sorted(self.hop.keys()):
            if G != (0, 0, 0):
                lines.extend(self._mat_to_hr(
                    G, self.hop[G]
                ))
        
        return '\n'.join(lines)

    def _mat_to_hr(self, G, mat):
        lines = []
        mat = np.array(mat).T # to be consistent with W90's ordering
        for j, column in enumerate(mat):
            for i, t in enumerate(column):
                lines.append(
                    '{0[0]:>5}{0[1]:>5}{0[2]:>5}{1:>5}{2:>5}{3.real:>12.6f}{3.imag:>12.6f}'.format(G, i + 1, j + 1, t)
                )
        return lines

    #-------------------MODIFYING THE MODEL ----------------------------#
    def add_hop(self, strength, orbital_1, orbital_2, rec_lattice_vec):
        """
        In all cases, the complex conjugate of the hopping is added automatically.

        .. warning:: This means that adding a hopping of strength :math:`\epsilon` between an orbital and itself in the home unit cell increases the orbitals on-site energy by :math:`2 \epsilon`.
        """
        G = tuple(rec_lattice_vec)
        mat = np.zeros((self.size, self.size), dtype=complex)
        if G == (0, 0, 0):
            mat[orbital_1, orbital_2] += strength
        elif G[np.nonzero(G)[0][0]] > 0:
            mat[orbital_1, orbital_2] += strength
        else:
            G = tuple(-x for x in G)
            mat[orbital_2, orbital_1] += strength.conjugate()
        self.hop[G] += sp.csr(mat)

    def add_on_site(self, energy, orbital):
        self.add_hop(energy / 2., orbital, orbital, (0, 0, 0))
        
    #-------------------CREATING DERIVED MODELS-------------------------#
    #---- arithmetic operations ----#
    def __add__(self, model):
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
        for G, hop_mat in model.hop.items():
            new_hop[G] += hop_mat
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
        return self + -model

    def __neg__(self):
        return -1 * self


    def __mul__(self, x):
        """
        Multiply on-site energies and hopping parameter strengths by a constant factor.
        """
        new_hop = dict()
        for G, hop_mat in self.hop.items():
            new_hop[G] = x * hop_mat

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
        return self * (1. / x)

    # for Python 3
    def __truediv__(self, x):
        return self.__div__(x)

    #---- other derived models ----#
    def supercell(self, dim, periodic=[True, True, True], passivation=None):
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
                    for G, hop_mat in self.hop.items():
                        hop_mat = np.array(hop_mat)
                        for i0, row in enumerate(hop_mat):
                            for i1, t in enumerate(row):
                                # new index of orbital 0
                                new_i0 = full_idx(uc0_pos, i0)
                                # position of the uc of orbital 1, not mapped inside supercell
                                full_uc1_pos = uc0_pos + np.array(G)
                                outside_supercell = [(p < 0) or (p >= d) for p, d in zip(full_uc1_pos, dim)]
                                # test if the hopping should be cut
                                cut_hop = any([not per and outside for per, outside in zip(periodic, outside_supercell)])
                                if cut_hop:
                                    continue
                                else:
                                    # G in terms of supercells
                                    new_G = np.array(np.floor(full_uc1_pos / dim), dtype=int)
                                    # mapped into the supercell
                                    uc1_pos = full_uc1_pos % dim
                                    new_i1 = full_idx(uc1_pos, i1)
                                    new_hop[tuple(new_G)][new_i0, new_i1] += t

        # new on_site terms, including passivation
        if passivation is None:
            passivation = lambda x, y, z: np.zeros(self.size)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    idx = (i * ny * nz + j * nz + k) * self.size
                    new_hop[(0, 0, 0)][idx:idx + self.size, idx:idx + self.size] += np.diag(np.array(passivation(*_edge_detect_pos([i, j, k], dim)), dtype=float) * 0.5)
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
        for G, hop in self.hop.items():
            if G not in new_hop.keys():
                new_hop[G] = np.zeros((2 * self.size, 2 * self.size), dtype=complex)
            new_hop[G][:self.size, :self.size] += hop
            # here you can either do -G  or hop.conjugate() or hop.T, but not both
            new_hop[G][self.size:, self.size:] += hop.conjugate()
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
        new_hop = {tuple(np.array(la.solve(uc, G), dtype=int)): hop_mat for G, hop_mat in self.hop.items()}

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
                    new_hop[(0, 0, 0)][i, i] += 0.5 * prefactor_scalar * scalar_pot(p)
                elif mode_scalar == 'absolute':
                    new_hop[(0, 0, 0)][i, i] += 0.5 * prefactor_scalar * scalar_pot(np.dot(self.uc, p))
                else:
                    raise ValueError('Unrecognized value for mode_scalar. Must be either "absolute" or "relative"')

        if vec_pot is not None:
            warnings.warn('The code for non-zero vector potential has not been tested at all!', UserWarning)
            vector_pot = lambda r: np.array(vec_pot(r))
            if self.uc is None:
                raise ValueError('Unit cell is not specified')
            for G, hop_mat in self.hop.items():
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
                    hop_mat[i0, i1] *= np.exp(-1j * prefactor_vec * np.dot(G + r1 - r0, A1 - A0))

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
