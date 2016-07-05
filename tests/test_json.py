#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    05.05.2015 13:59:00 CEST
# File:    hr_hamilton.py

import pytest

import tbmodels
import numpy as np

from models import get_model

kpt = [(0.1, 0.2, 0.7), (-0.3, 0.5, 0.2), (0., 0., 0.), (0.1, -0.9, -0.7)]


@pytest.mark.parametrize('kwargs', [
    dict(),
    dict(pos=None, dim=3),
    dict(uc=3 * np.eye(3)),
    dict(pos=np.zeros((2, 3)), uc=np.eye(3))
])
def test_json_consistency(get_model, models_equal, kwargs):
    model1 = get_model(0.1, 0.2, **kwargs)
    model2 = tbmodels.Model.from_json(model1.to_json())
    models_equal(model1, model2)
