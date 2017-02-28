#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>

import numpy as np
import matplotlib.pyplot as plt

import tbmodels as tb
from tbmodels.helpers import SymmetryOperation, Representation

if __name__ == '__main__':
    HDF5_FILE = 'data/model_nosym.hdf5'
    try:
        model = tb.Model.from_hdf5_file(HDF5_FILE)
    except OSError:
        model = tb.Model.from_wannier_files(
            hr_file='data/wannier90_hr.dat',
            pos=[(0, 0, 0)] * 8 + [(0.5, 0.5, 0.5)] * 6,
            occ=6
        )
        model.to_hdf5_file(HDF5_FILE)

    # model = model.slice_orbitals([0, 2, 3, 1, 5, 6, 4, 7, 9, 10, 8, 12, 13, 11])
    model = model.slice_orbitals([0, 2, 3, 1, 5, 6, 4, 7, 9, 10, 8, 12, 13, 11])

    # set up symmetry operations
    time_reversal = SymmetryOperation(
        kmatrix = -np.eye(3),
        repr=Representation(
            complex_conjugate=True,
            matrix=np.kron([[0, -1j], [1j, 0]], np.eye(7))
        )
    )

    print(model.eigenval((0, 0, 0.1)))
    model = model.symmetrize([time_reversal])
    print(model.eigenval((0, 0, 0.1)))

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
