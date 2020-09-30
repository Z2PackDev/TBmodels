#!/usr/bin/env python

# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Test the conversion to / from the pickle format.
"""

import pickle

import pytest
import numpy as np

KWARGS = [
    dict(),
    dict(pos=None, dim=3),
    dict(uc=3 * np.eye(3)),
    dict(pos=np.zeros((2, 3)), uc=np.eye(3)),
]


@pytest.mark.parametrize("kwargs", KWARGS)
def test_pickle_consistency(get_model, models_equal, kwargs):
    """
    Check that a simple model remains unchanged after passing through
    pickle.
    """
    model1 = get_model(0.1, 0.2, **kwargs)
    model2 = pickle.loads(pickle.dumps(model1))
    models_equal(model1, model2)
