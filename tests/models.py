#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    15.10.2014 10:18:11 CEST
# File:    common.py

import itertools
from collections import ChainMap

import pytest
import tbmodels

@pytest.fixture(params=[True, False]) # params is for sparse / dense
def get_model(request):
    def inner(t1, t2, **kwargs):
        dim = kwargs.get('dim', 3)
        defaults = {}
        defaults['pos'] = [[0] * 2, [0.5] * 2]
        if dim < 2:
            raise ValueError('dimension must be at least 2')
        elif dim > 2:
            for p in defaults['pos']:
                p.extend([0] * (dim - 2))
        defaults['occ'] = 1
        defaults['on_site'] = (1, -1)
        defaults['size'] = 2
        defaults['dim'] = None
        defaults['sparse'] = request.param
        model = tbmodels.Model(**ChainMap(kwargs, defaults))

        for phase, R in zip([1, -1j, 1j, -1], itertools.product([0, -1], [0, -1], [0])):
            model.add_hop(t1 * phase, 0, 1, R)

        for R in ((r[0], r[1], 0) for r in itertools.permutations([0, 1])):
            model.add_hop(t2, 0, 0, R)
            model.add_hop(-t2, 1, 1, R)
        return model
    return inner
