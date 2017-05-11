#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    05.07.2016 17:03:24 CEST
# File:    test_json_encoding.py

import json

import pytest
import numpy as np

from tbmodels.helpers import encode, decode

@pytest.mark.parametrize('obj', ['test_string', True, False, np.bool_(True), None, {'a': 2, 'b': 3}])
def test_dumpload(obj):
    assert json.loads(json.dumps(obj, default=encode), object_hook=decode) == obj

@pytest.mark.parametrize('obj,res', [((1, 2, 3), [1, 2, 3])])
def test_dumpload_changed(obj, res):
    assert json.loads(json.dumps(obj, default=encode), object_hook=decode) == res

@pytest.mark.parametrize('obj', [lambda x: True])
def test_fail(obj):
    with pytest.raises(TypeError):
        json.loads(json.dumps(obj, default=encode), object_hook=decode)
