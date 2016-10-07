#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    08.04.2016 13:58:36 CEST
# File:    test_doc.py

import sys
import pytest
import importlib

import fsc.export

# This should never appear in any serious code ;)
# To out-manoeuver pickle's caching, and force re-loading tbmodels
def test_all_doc():
    old_name = 'tbmodels'
    new_name = 'hoopy_tbmodels'
    for key in list(sys.modules.keys()):
        # move previous tbmodels to hoopy_tbmodels
        if key.startswith(old_name):
            new_key = key.replace(old_name, new_name)
            sys.modules[new_key] = sys.modules[key]
            del sys.modules[key]
    fsc.export.test_doc()
    try:
        import tbmodels
    finally:
        # reset to the previous tbmodels -- just doing import breaks pickle
        for key in list(sys.modules.keys()):
            if key.startswith(old_name):
                del sys.modules[key]
        for key in list(sys.modules.keys()):
            if key.startswith(new_name):
                new_key = key.replace(new_name, old_name)
                sys.modules[new_key] = sys.modules[key]
                del sys.modules[key]
