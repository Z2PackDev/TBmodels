#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the method of removing small hopping terms."""

import pytest


@pytest.mark.parametrize('t1', [0.6 + 0.5j, 0. + 1j, 2])
@pytest.mark.parametrize('t2', [0.2j, 1.5])
@pytest.mark.parametrize('onsite', [(1.2, -0.6), (3.5, 2)])
@pytest.mark.parametrize('cutoff', [-1, 0.4, 1, 3, 10])
def test_simple(t1, t2, onsite, cutoff, get_model, models_equal):
    """
    Check that cutting a small hopping produced the same model as
    constructing the model without that hopping.
    """
    def _truncate(val):
        print(val)
        print(val if abs(val) >= cutoff else 0.)
        return val if abs(val) >= cutoff else 0.

    model = get_model(t1, t2, on_site=onsite)
    model.remove_small_hop(cutoff=cutoff)
    reference = get_model(
        _truncate(t1), _truncate(t2), on_site=(_truncate(onsite[0]), _truncate(onsite[1]))
    )
    models_equal(model, reference)
