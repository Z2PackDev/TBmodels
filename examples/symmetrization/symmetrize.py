#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Georg Winkler, Dominik Gresch <greschd@gmx.ch>

import random

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import tbmodels as tb
from tbmodels.helpers import SymmetryOperation, Representation

import pymatgen as mg

if __name__ == '__main__':
    HDF5_FILE = 'data/model_nosym.hdf5'
    try:
        model_nosym = tb.Model.from_hdf5_file(HDF5_FILE)
    except OSError:
        model_nosym = tb.Model.from_wannier_files(
            hr_file='data/wannier90_hr.dat',
            pos=([(0, 0, 0)] * 4 + [(0.25, 0.25, 0.25)] * 3) * 2,
            occ=6
        )
        model_nosym.to_hdf5_file(HDF5_FILE)

    model_nosym = model_nosym.slice_orbitals([0, 2, 3, 1, 5, 6, 4, 7, 9, 10, 8, 12, 13, 11])

    # set up symmetry operations
    time_reversal = SymmetryOperation(
        kmatrix = -np.eye(3),
        repr=Representation(
            complex_conjugate=True,
            matrix=np.kron([[0, -1j], [1j, 0]], np.eye(7))
        )
    )

    def spin_reps(prep):
        """
        Calculates the spin rotation matrices. The formulas to determine the rotation axes and angles
        are taken from `here <http://scipp.ucsc.edu/~haber/ph116A/rotation_11.pdf>`_.

        :param prep:   List that contains 3d rotation matrices.
        :type prep:    list(array)
        """
        # general representation of the D1/2 rotation about the axis (l,m,n) around the
        # angle phi
        D12 = lambda l, m, n, phi: np.array([[np.cos(phi / 2.) - 1j * n * np.sin(phi / 2.), (-1j * l - m) * np.sin(phi / 2.)],
                                             [(-1j * l + m) * np.sin(phi / 2.), np.cos(phi / 2.) + 1j * n * np.sin(phi / 2.)]])

        n = np.zeros(3)
        tr = np.trace(prep)
        det = np.round(np.linalg.det(prep), 5)
        if det == 1.:  # rotations
            theta = np.arccos(0.5 * (tr - 1.))
            if theta != 0:
                n[0] = prep[2, 1] - prep[1, 2]
                n[1] = prep[0, 2] - prep[2, 0]
                n[2] = prep[1, 0] - prep[0, 1]
                if np.round(np.linalg.norm(n), 5) == 0.:  # theta = pi, that is C2 rotations
                    e, v = la.eig(prep)
                    n = v[:, list(np.round(e, 10)).index(1.)]
                    spin = np.round(D12(n[0], n[1], n[2], np.pi), 15)
                else:
                    n /= np.linalg.norm(n)
                    spin = np.round(D12(n[0], n[1], n[2], theta), 15)
            else:  # case of unitiy
                spin = D12(0, 0, 0, 0)
        elif det == -1.:  # improper rotations and reflections
            theta = np.arccos(0.5 * (tr + 1.))
            if np.round(theta, 5) != np.round(np.pi, 5):
                n[0] = prep[2, 1] - prep[1, 2]
                n[1] = prep[0, 2] - prep[2, 0]
                n[2] = prep[1, 0] - prep[0, 1]
                if np.round(np.linalg.norm(n), 5) == 0.:  # theta = 0 (reflection)
                    e, v = la.eig(prep)
                    # normal vector is eigenvector to eigenvalue -1
                    n = v[:, list(np.round(e, 10)).index(-1.)]
                    # spin is a pseudovector!
                    spin = np.round(D12(n[0], n[1], n[2], np.pi), 15)
                else:
                    n /= np.linalg.norm(n)
                    # rotation followed by reflection:
                    spin = np.round(
                        np.dot(D12(n[0], n[1], n[2], np.pi), D12(n[0], n[1], n[2], theta)), 15)
            else:  # case of inversion (does not do anything to spin)
                spin = D12(0, 0, 0, 0)
        return np.array(spin)

    structure = mg.Structure.from_file('data/POSCAR')

    # get real-space representations
    analyzer = mg.symmetry.analyzer.SpacegroupAnalyzer(structure)
    symops = analyzer.get_symmetry_operations(cartesian=False)
    symops_cart = analyzer.get_symmetry_operations(cartesian=True)
    rots = [x.rotation_matrix for x in symops]
    taus = [x.translation_vector for x in symops]

    # print(symops)
    # sys.exit()

    # get corresponding represesentations in the Hamiltonian basis
    reps = []
    for n, (rot, tau) in enumerate(zip(rots, taus)):
        C = symops_cart[n].rotation_matrix
        tauc = symops_cart[n].translation_vector
        prep = C
        spinrep = spin_reps(C).conj()
        R = np.kron(spinrep, la.block_diag(1., prep, prep))
        reps.append(R)

    symmetries = [
        SymmetryOperation(
            kmatrix=rot.transpose(),
            repr=Representation(
                complex_conjugate=False,
                matrix=repr_mat.conj()
            )
        )
        for rot, repr_mat in zip(rots, reps)
    ]

    model = model_nosym.symmetrize([time_reversal] + symmetries)

    # for R in reps:
    #     assert np.isclose(R.conjugate().transpose(), la.inv(R)).all()

    # print(model.eigenval(k))
    # print(model_nosym.eigenval(k))
    # assert np.isclose(model_nosym.eigenval(k), model.eigenval(k), atol=1e-3).all()

    # efermi = la.eigh(tbm.H([0, 0, 0]))[0][tbm.nocc]
    # nsteps = 50
    # path = getpath(np.array([L, G, X, L, W, G, U]), nsteps, b)
    # dist = getdist(path)
    # E = calcEpath(path, tbm_SO, verbose=True)
    # # E1 = calcEpath(path,Hsym,verbose=True)
    # E1 = calcEpath(path, tbm_sym_SO, verbose=True)
    #
    # plt.figure()
    # plt.plot(dist, E - efermi, 'k')
    # plt.plot(dist, E1 - efermi, 'r--')
    # for x in dist[::nsteps]:
    #     plt.axvline(x, color='b')
    # plt.xticks(dist[::nsteps], ['L', 'G', 'X', 'L', 'W', 'G', 'U'])
    # plt.xlim([dist[0], dist[-1]])
    # plt.ylim([-6, 6])
    # plt.savefig(folder + 'compare_bands.pdf')
