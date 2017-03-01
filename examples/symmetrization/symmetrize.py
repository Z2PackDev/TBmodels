#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Georg Winkler, Dominik Gresch <greschd@gmx.ch>

import sys
import random

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import tbmodels as tb
from tbmodels.helpers import SymmetryOperation, Representation

import pymatgen as mg
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def getpath(P, n, b=None):
    """ Calculates path in k-space given symmetry points P, length of
    a segment n and reciprocal lattice vectors b """
    P = np.array(P)
    if b is None:
        b = np.eye(P.shape[1])
    l = len(P)
    path = np.zeros(((l - 1) * n + 1, P.shape[1]))
    for i in range(0, l - 1):
        for j in range(0, n):
            p = P[i, :] + (P[i + 1, :] - P[i, :]) * float(j) / (n)
            path[i * n + j, :] = np.dot(p, b)
    path[-1, :] = np.dot(P[-1], b)
    return path


def getdist(path, b=None):
    """ Calculates distance from a k-path """
    dist = np.zeros(len(path))
    if b is None:
        for i in range(1, len(path)):
            dist[i] = dist[i - 1] + la.norm(path[i] - path[i - 1])
    else:
        for i in range(1, len(path)):
            dist[i] = dist[i - 1] + la.norm(dot(path[i] - path[i - 1], b))
    return dist

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

if __name__ == '__main__':
    HDF5_FILE = 'data/model_nosym.hdf5'
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
    analyzer = SpacegroupAnalyzer(structure)
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

    symmetries = [
        SymmetryOperation(
            kmatrix=rot.transpose(),
            repr=Representation(
                complex_conjugate=False,
                matrix=repr_mat
            )
        )
        for rot, repr_mat in zip(rots, reps)
    ]

    model = model_nosym.symmetrize([time_reversal] + symmetries, full_group=True)

    # band structure
    G = np.array([0., 0., 0.])
    L = np.array([0.5, 0.5, 0.5])
    X = np.array([.5, 0., 0.5])
    W = np.array([0.5, 0.25, 0.75])
    U = np.array([0.25, 0.625, 0.625])

    efermi = model.eigenval([0, 0, 0])[model.occ]
    nsteps = 100
    path = getpath(np.array([L, G, X, L, W, G, U]), nsteps)
    dist = getdist(path)
    E = [model_nosym.eigenval(k) for k in path]
    E1 = [model.eigenval(k) for k in path]

    plt.figure()
    for x in dist[::nsteps]:
        plt.axvline(x, color='b')
    plt.plot(dist, E - efermi, 'k')
    plt.plot(dist, E1 - efermi, 'r--')
    plt.xticks(dist[::nsteps], ['L', 'G', 'X', 'L', 'W', 'G', 'U'])
    plt.xlim([dist[0], dist[-1]])
    plt.ylim([-6, 6])
    plt.savefig('plots/compare_bands.pdf')
