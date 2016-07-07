#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    07.07.2016 00:46:23 CEST
# File:    test.py

import itertools

import tbmodels
import numpy as np
from tbmodels._ptools.monitoring import Timer

k_list = list(itertools.product(np.linspace(0, 1, 5), repeat=3))

model2 = tbmodels.Model.from_hr_file('data/wannier90_hr.dat', sparse=False)
with Timer('dense'):
    for k in k_list:
        model2.hamilton(k)

model1 = tbmodels.Model.from_hr_file('data/wannier90_hr.dat')
with Timer('sparse'):
    for k in k_list:
        model1.hamilton(k)
