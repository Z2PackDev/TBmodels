#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    04.07.2016 15:42:15 CEST
# File:    test_matrix_to_hoppings.py

import pytest

import tbmodels
import numpy as np

from models import get_model

def test_consistency(get_model, models_equal):
    model1 = get_model(0.1, 0.2)
    
    hoppings = []
    for k, v in model1.hop.items():
        hoppings.extend(tbmodels.helpers.matrix_to_hop(np.array(v), R=k))
        
    model2 = tbmodels.Model.from_hop_list(size=2, hop_list=hoppings, contains_cc=False, occ=1, pos=((0.,) * 3, (0.5, 0.5, 0.)))
    models_equal(model1, model2)
