#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Authors: Georg Winkler, Dominik Gresch <greschd@gmx.ch>

import os

import numpy as np
import matplotlib.pyplot as plt
import tbmodels as tb
import pymatgen as mg
import pymatgen.symmetry.analyzer
import pymatgen.symmetry.bandstructure
import symmetry_representation as sr


def compare_bands_plot(*models, structure):
    path = mg.symmetry.bandstructure.HighSymmKpath(structure)
    kpts, labels = path.get_kpoints(line_density=100)
    # de-duplicate / merge labels
    for i in range(len(labels) - 1):
        if labels[i] and labels[i + 1]:
            if labels[i] != labels[i + 1]:
                labels[i] = labels[i] + ' | ' + labels[i + 1]
            labels[i + 1] = ''

    energies = []
    for m in models:
        energies.append([m.eigenval(k) for k in kpts])

    plt.figure()
    labels_clean = []
    labels_idx = []
    for i, l in enumerate(labels):
        if l:
            labels_idx.append(i)
            labels_clean.append('$' + l + '$')
    for i in labels_idx[1:-1]:
        plt.axvline(i, color='k', lw=0.8)

    lengths = np.linspace(1.2, 0.6, len(energies), endpoint=True)
    for i, (e, lw) in enumerate(zip(energies, lengths)):
        plt.plot(range(len(kpts)), e, color=f'C{i}', lw=lw)
    plt.xticks(labels_idx, labels_clean)
    plt.xlim([0, len(kpts) - 1])
    plt.savefig('results/compare_bands.pdf', bbox_inches='tight')


if __name__ == '__main__':
    model_nosym = tb.Model.from_hdf5_file('data/model_nosym.hdf5')

    structure = mg.Structure(
        lattice=model_nosym.uc, species=['Si', 'Si'], coords=np.array([[0.5, 0.5, 0.5], [0.75, 0.75, 0.75]])
    )

    orbitals = [
        sr.Orbital(position=coord, function_string=fct, spin=spin) for spin in (sr.SPIN_UP, sr.SPIN_DOWN)
        for coord in ([0.5, 0.5, 0.5], [0.75, 0.75, 0.75]) for fct in sr.WANNIER_ORBITALS['sp3']
    ]

    time_reversal = sr.get_time_reversal(orbitals=orbitals, numeric=True)

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
    model.to_hdf5_file('results/model.hdf5')

    compare_bands_plot(model_nosym, model, structure=structure)
