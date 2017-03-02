#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Georg Winkler, Dominik Gresch <greschd@gmx.ch>

import os
import sys
import random

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import tbmodels as tb
from tbmodels.helpers import SymmetryOperation, Representation

import pymatgen as mg
import pymatgen.symmetry.analyzer
import pymatgen.symmetry.bandstructure

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

def compare_bands_plot(model1, model2, structure):
    path = mg.symmetry.bandstructure.HighSymmKpath(structure)
    kpts, labels = path.get_kpoints(line_density=50)
    # de-duplicate / merge labels
    for i in range(len(labels) - 1):
        if labels[i] and labels[i + 1]:
            if labels[i] != labels[i + 1]:
                labels[i] = labels[i] + ' | ' + labels[i + 1]
            labels[i + 1] = ''

    efermi = model1.eigenval([0, 0, 0])[model1.occ]
    E = [model2.eigenval(k) for k in kpts]
    E_sym = [model1.eigenval(k) for k in kpts]

    plt.figure()
    labels_clean = []
    labels_idx = []
    for i, l in enumerate(labels):
        if l:
            labels_idx.append(i)
            labels_clean.append('$' + l + '$')
    for i in labels_idx[1:-1]:
        plt.axvline(i, color='b')
    plt.plot(range(len(kpts)), E - efermi, 'k')
    plt.plot(range(len(kpts)), E_sym - efermi, 'r--')
    plt.xticks(labels_idx, labels_clean)
    plt.xlim([0, len(kpts) - 1])
    plt.ylim([-6, 6])
    plt.savefig('results/compare_bands.pdf')

if __name__ == '__main__':
    HDF5_FILE = 'results/model_nosym.hdf5'
    try:
        model_nosym = tb.Model.from_hdf5_file(HDF5_FILE)
    except OSError:
        model_nosym = tb.Model.from_wannier_files(
            hr_file='data/wannier90_hr.dat',
            win_file='data/wannier90.win',
            pos=([(0, 0, 0)] * 4 + [(0.25, 0.25, 0.25)] * 3) * 2,
            occ=6
        )
        model_nosym.to_hdf5_file(HDF5_FILE)

    # change the order of the orbitals from (In: s, py, pz, px; As: py, pz, px) * 2
    # to (In: s, px, py, pz; As: s, px, py, pz) * 2
    model_nosym = model_nosym.slice_orbitals([0, 2, 3, 1, 5, 6, 4, 7, 9, 10, 8, 12, 13, 11])

    # set up symmetry operations
    time_reversal = SymmetryOperation(
        kmatrix = -np.eye(3),
        repr=Representation(
            complex_conjugate=True,
            matrix=np.kron([[0, -1j], [1j, 0]], np.eye(7))
        )
    )

    structure = mg.Structure(
        lattice=model_nosym.uc,
        species=['In', 'As'],
        coords=np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
    )

    # get real-space representations
    analyzer = mg.symmetry.analyzer.SpacegroupAnalyzer(structure)
    symops = analyzer.get_symmetry_operations(cartesian=False)
    symops_cart = analyzer.get_symmetry_operations(cartesian=True)
    rots = [x.rotation_matrix for x in symops]
    taus = [x.translation_vector for x in symops]

    # get corresponding represesentations in the Hamiltonian basis
    reps = []
    for n, (rot, tau) in enumerate(zip(rots, taus)):
        C = symops_cart[n].rotation_matrix
        tauc = symops_cart[n].translation_vector
        prep = C
        spinrep = spin_reps(C)
        R = np.kron(spinrep, la.block_diag(1., prep, prep))
        reps.append(R)

    # set up the space group symmetries
    symmetries = [
        SymmetryOperation(
            # r-space and k-space matrices are related by transposing
            kmatrix=rot.transpose(),
            repr=Representation(
                complex_conjugate=False,
                matrix=repr_mat
            )
        )
        for rot, repr_mat in zip(rots, reps)
    ]

    os.makedirs('results', exist_ok=True)
    # model = model_nosym.symmetrize([time_reversal] + symmetries, full_group=True)
    model_tr = model_nosym.symmetrize([time_reversal])
    model = model_tr
    # model = model_tr.symmetrize(symmetries, full_group=True)
    model.to_hdf5_file('results/model.hdf5')

    compare_bands_plot(model, model_nosym, structure)

    for k in [(0., 0., 0.), (0.12312351, 0.73475412, 0.2451235)]:
        A = model.hamilton(k, convention=1)
        B = time_reversal.repr.matrix @ model.hamilton(la.inv(time_reversal.kmatrix) @ k, convention=1).conjugate() @ time_reversal.repr.matrix.conjugate().transpose()
        print(np.isclose(A, B).all())
        print(np.max(np.abs(A - B)))
        # print(model.hamilton(k))
        # print(time_reversal.repr.matrix @ model.hamilton(time_reversal.kmatrix @ k).conjugate() @ time_reversal.repr.matrix.conjugate().transpose())
        # for sym in symmetries:
        #     print(np.isclose(
        #         model.hamilton(k, convention=1),
        #         sym.repr.matrix.conjugate().transpose() @ model.hamilton(la.inv(sym.kmatrix) @ k, convention=1) @ sym.repr.matrix
        #     ).all())
