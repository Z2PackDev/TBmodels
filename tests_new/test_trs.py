#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    15.10.2014 10:22:43 CEST
# File:    trs.py

import pytest

import tbmodels
import numpy as np

from models import get_model

def test_trs(get_model, compare_data):
    model = get_model(0.2, 0.3)
    trs_model = model.trs()
    compare_data(
        lambda x, y: np.isclose(x, y).all(),
        trs_model.hamilton([0.4, -0.2, 0.1])
    )
