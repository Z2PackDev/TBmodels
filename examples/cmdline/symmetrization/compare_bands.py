#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Georg Winkler, Dominik Gresch <greschd@gmx.ch>

import tbmodels
import numpy as np
import matplotlib.pyplot as plt
import pymatgen as mg
import pymatgen.symmetry.bandstructure

def compare_bands_plot(model1, model2, structure):
    path = mg.symmetry.bandstructure.HighSymmKpath(structure)
    kpts, labels = path.get_kpoints(line_density=200)
    # de-duplicate / merge labels
    for i in range(len(labels) - 1):
        if labels[i] and labels[i + 1]:
            if labels[i] != labels[i + 1]:
                labels[i] = labels[i] + ' | ' + labels[i + 1]
            labels[i + 1] = ''

    # E-fermi is just an approximation
    efermi = model1.eigenval([0, 0, 0])[model1.occ]
    E1 = [model1.eigenval(k) for k in kpts]
    E2 = [model2.eigenval(k) for k in kpts]

    plt.figure()
    labels_clean = []
    labels_idx = []
    for i, l in enumerate(labels):
        if l:
            labels_idx.append(i)
            labels_clean.append('$' + l + '$')
    for i in labels_idx[1:-1]:
        plt.axvline(i, color='b')
    plt.plot(range(len(kpts)), E1 - efermi, 'k')
    plt.plot(range(len(kpts)), E2 - efermi, 'r', lw=0.5)
    plt.xticks(labels_idx, labels_clean)
    plt.xlim([0, len(kpts) - 1])
    plt.ylim([-6, 6])
    plt.savefig('results/compare_bands.pdf', bbox_inches='tight')

if __name__ == '__main__':
    model_nosym = tbmodels.Model.from_hdf5_file('data/model_nosym.hdf5')
    model_sym = tbmodels.Model.from_hdf5_file('results/model_symmetrized.hdf5')
    reference_model = tbmodels.Model.from_hdf5_file('data/reference_model.hdf5')

    structure = mg.Structure(
        lattice=model_nosym.uc,
        species=['In', 'As'],
        coords=np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
    )

    compare_bands_plot(model_nosym, model_sym, structure)
    for R in set(model_sym.hop.keys()) | set(reference_model.hop.keys()):
        assert np.isclose(model_sym.hop[R], reference_model.hop[R]).all()

    # Check that the symmetries are fulfilled at some random k
    k = (0.12312351, 0.73475412, 0.2451235)
    assert np.isclose(
        model_sym.hamilton(k, convention=1),
        time_reversal.repr.matrix @
        # when complex conjugation is present, r-space matrix (R) and k-space matrix (K)
        # are related by K = -(R.T)^{-1}
        # -> K^{-1} = -R.T
        model_sym.hamilton(-time_reversal.rotation_matrix.T @ k, convention=1).conjugate() @
        time_reversal.repr.matrix.conjugate().T
    ).all()

    for sym in symmetries:
        assert np.isclose(
            model_sym.hamilton(k, convention=1),
            sym.repr.matrix @
            # k-space and r-space matrices are related by transposing and inverting
            # -> k-matrix^{-1} == r-matrix.T
            model_sym.hamilton(sym.rotation_matrix.T @ k, convention=1) @
            sym.repr.matrix.conjugate().T
        ).all()
