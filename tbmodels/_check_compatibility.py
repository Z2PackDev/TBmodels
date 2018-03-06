"""
Utilities to check whether models are compatible, e.g. for adding.
"""


def _check_dim(*models):
    return len(set([m.dim for m in models])) == 1


def _check_uc(*models):
    if any(m.uc is None for m in models):
        return all(m.uc is None for m in models)
    tolerance = 1e-6
    reference_uc = models[0].uc
    for m in models:
        for v1, v2 in zip(reference_uc, m.uc):
            for x1, x2 in zip(v1, v2):
                if abs(x1 - x2) > tolerance:
                    return False
    return True
