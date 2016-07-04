#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    04.07.2016 14:01:18 CEST
# File:    test_invalid_constructors.py

import pytest
import tbmodels

from models import get_model

def test_on_site_too_long(get_model):
    with pytest.raises(ValueError):
        get_model(0.1, 0.2, on_site=[1, 2, 3])

def test_no_size_given(get_model, models_equal):
    model1 = get_model(0.1, 0.2, size=None)
    model2 = get_model(0.1, 0.2)
    models_equal(model1, model2)
    
def test_size_unknown(get_model):
    with pytest.raises(ValueError):
        get_model(0.1, 0.2, size=None, on_site=None)
