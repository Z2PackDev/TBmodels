#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Georg Winkler, Dominik Gresch <greschd@gmx.ch>

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
