#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    30.06.2016 15:11:19 CEST
# File:    parameters.py

from os.path import abspath, dirname, join

T_VALUES = [(t1, t2) for t1 in [-0.1, 0.2, 0.3] for t2 in [-0.2, 0.5]]
KPT = [(0.1, 0.2, 0.7), (-0.3, 0.5, 0.2), (0., 0., 0.), (0.1, -0.9, -0.7)]
SAMPLES_DIR = join(dirname(abspath(__file__)), 'samples')
