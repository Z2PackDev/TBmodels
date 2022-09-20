# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""Utilities to check whether models are compatible, e.g. for adding."""


def check_dim(*models):
    """Check that the dimension of all models is the same."""
    return len({m.dim for m in models}) == 1


def check_uc(*models):
    """Check if the unit cells of two models are approximately the same."""
    if any(m.uc is None for m in models):
        return all(m.uc is None for m in models)
    tolerance = 1e-6
    reference_uc = models[0].uc
    for m in models:
        for vec1, vec2 in zip(reference_uc, m.uc):
            for x1, x2 in zip(vec1, vec2):
                if abs(x1 - x2) > tolerance:
                    return False
    return True
