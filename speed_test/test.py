#!/usr/bin/env python
# -*- coding: utf-8 -*-

# © 2015-2018, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import itertools

import tbmodels
import numpy as np
from tbmodels._ptools.monitoring import Timer

k_list = list(itertools.product(np.linspace(0, 1, 5), repeat=3))

print('dense model')

model2 = tbmodels.Model.from_hr_file('data/dense_hr.dat', sparse=False)
with Timer('dense'):
    for k in k_list:
        model2.hamilton(k)

model1 = tbmodels.Model.from_hr_file('data/dense_hr.dat', sparse=True)
with Timer('sparse'):
    for k in k_list:
        model1.hamilton(k)

# For the sparse model a supercell or TRS model should be used
#~ print('sparse model')
#~ model2 = tbmodels.Model.from_hr_file('data/sparse_hr.dat', sparse=False)
#~ with Timer('dense'):
#~ for k in k_list:
#~ model2.hamilton(k)

#~ model1 = tbmodels.Model.from_hr_file('data/sparse_hr.dat', sparse=True)
#~ with Timer('sparse'):
#~ for k in k_list:
#~ model1.hamilton(k)
