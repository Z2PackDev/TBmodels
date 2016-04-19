#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    15.10.2014 10:18:11 CEST
# File:    common.py

import pytest
import tbmodels

@pytest.fixture
def get_model():
    def inner(t1, t2, dim=3, uc=None):
        pos = [[0] * 2, [0.5] * 2]
        if dim < 2:
            raise ValueError('dimension must be at least 2')
        elif dim > 2:
            for p in pos:
                p.extend([0] * (dim - 2))
        model = tbmodels.Model(size=2, on_site=[1, -1], pos=pos, occ=1, uc=uc)

        for phase, R in zip([1, -1j, 1j, -1], tbmodels.helpers.combine([0, -1], [0, -1], 0)):
            model.add_hop(t1 * phase, 0, 1, R)

        for R in tbmodels.helpers.neighbours([0, 1], forward_only=True):
            model.add_hop(t2, 0, 0, R)
            model.add_hop(-t2, 1, 1, R)
        return model
    return inner
