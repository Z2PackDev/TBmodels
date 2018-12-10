#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Georg Winkler, Dominik Gresch <greschd@gmx.ch>

import os

import numpy as np
import matplotlib.pyplot as plt
import tbmodels as tb
import pymatgen as mg
import pymatgen.symmetry.analyzer
import pymatgen.symmetry.bandstructure
import symmetry_representation as sr


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
    plt.savefig('results/compare_bands_new.pdf', bbox_inches='tight')


if __name__ == '__main__':
    model_nosym = tb.Model.from_hdf5_file('data/model_nosym.hdf5')
    reference_model = tb.Model.from_hdf5_file('data/reference_model.hdf5')

    # change the order of the orbitals from (In: s, py, pz, px; As: py, pz, px) * 2
    # to (In: s, px, py, pz; As: px, py, pz) * 2
    model_nosym = model_nosym.slice_orbitals([0, 2, 3, 1, 5, 6, 4, 7, 9, 10, 8, 12, 13, 11])

    pos_In = (0, 0, 0)
    pos_As = (0.25, 0.25, 0.25)
    spin_up = sr.Spin(total=0.5, z_component=0.5)
    spin_down = sr.Spin(total=0.5, z_component=-0.5)
    orbitals = [
        sr.Orbital(position=pos_In, function_string='1', spin=spin_up),
        sr.Orbital(position=pos_In, function_string='x', spin=spin_up),
        sr.Orbital(position=pos_In, function_string='y', spin=spin_up),
        sr.Orbital(position=pos_In, function_string='z', spin=spin_up),
        sr.Orbital(position=pos_As, function_string='x', spin=spin_up),
        sr.Orbital(position=pos_As, function_string='y', spin=spin_up),
        sr.Orbital(position=pos_As, function_string='z', spin=spin_up),
        sr.Orbital(position=pos_In, function_string='1', spin=spin_down),
        sr.Orbital(position=pos_In, function_string='x', spin=spin_down),
        sr.Orbital(position=pos_In, function_string='y', spin=spin_down),
        sr.Orbital(position=pos_In, function_string='z', spin=spin_down),
        sr.Orbital(position=pos_As, function_string='x', spin=spin_down),
        sr.Orbital(position=pos_As, function_string='y', spin=spin_down),
        sr.Orbital(position=pos_As, function_string='z', spin=spin_down),
    ]

    # set up symmetry operations
    time_reversal = sr.get_time_reversal(orbitals=orbitals, numeric=True)
    assert np.allclose(time_reversal.repr.matrix, np.kron([[0, -1j], [1j, 0]], np.eye(7)))

    structure = mg.Structure(
        lattice=model_nosym.uc, species=['In', 'As'], coords=np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
    )

    # get real-space representations
    analyzer = mg.symmetry.analyzer.SpacegroupAnalyzer(structure)
    symops = analyzer.get_symmetry_operations(cartesian=False)
    symops_cart = analyzer.get_symmetry_operations(cartesian=True)

    symmetries = []
    for sym, sym_cart in zip(symops, symops_cart):
        symmetries.append(
            sr.SymmetryOperation.from_orbitals(
                orbitals=orbitals,
                real_space_operator=sr.RealSpaceOperator.from_pymatgen(sym),
                rotation_matrix_cartesian=sym_cart.rotation_matrix,
                numeric=True
            )
        )

    os.makedirs('results', exist_ok=True)
    model_tr = model_nosym.symmetrize([time_reversal])
    model = model_tr.symmetrize(symmetries, full_group=True)
    model.to_hdf5_file('results/model_new.hdf5')

    compare_bands_plot(model_nosym, model, structure)
    for R in set(model.hop.keys()) | set(reference_model.hop.keys()):
        assert np.isclose(model.hop[R], reference_model.hop[R]).all()

    # Check that the symmetries are fulfilled at some random k
    k = (0.12312351, 0.73475412, 0.2451235)
    assert np.isclose(
        model.hamilton(k, convention=1),
        time_reversal.repr.matrix @
        # when complex conjugation is present, r-space matrix (R) and k-space matrix (K)
        # are related by K = -(R.T)^{-1}
        # -> K^{-1} = -R.T
        model.hamilton(-time_reversal.rotation_matrix.T @ k, convention=1).conjugate() @
        time_reversal.repr.matrix.conjugate().T
    ).all()

    for sym in symmetries:
        assert np.isclose(
            model.hamilton(k, convention=1),
            sym.repr.matrix @
            # k-space and r-space matrices are related by transposing and inverting
            # -> k-matrix^{-1} == r-matrix.T
            model.hamilton(sym.rotation_matrix.T @ k, convention=1) @
            sym.repr.matrix.conjugate().T
        ).all()
