#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import pytest

from parameters import T_VALUES, KPT


@pytest.mark.parametrize('t1', T_VALUES)
@pytest.mark.parametrize('k', KPT)
def test_simple(t1, get_model, k, compare_data, models_equal, compare_isclose):
    m = get_model(*t1)

    compare_isclose(m.hamilton(k), tag='hamilton')
    compare_isclose(m.eigenval(k), tag='eigenval')
    compare_data(models_equal, m)


@pytest.mark.parametrize('t1', T_VALUES)
def test_invalid_R(t1, get_model):
    with pytest.raises(ValueError):
        m = get_model(*t1, dim=2)
