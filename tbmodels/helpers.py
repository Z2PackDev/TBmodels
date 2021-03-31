#
# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
This module contains a helper function to create a list of hoppings from a given matrix (:meth:`matrix_to_hop`).
"""

import typing as ty

import numpy as np
from fsc.export import export


@export
def matrix_to_hop(
    mat: ty.Sequence[ty.Sequence[complex]],
    orbitals: ty.Optional[ty.Sequence[int]] = None,
    R: ty.Sequence[int] = (0, 0, 0),
    multiplier: float = 1.0,
) -> ty.List[ty.Tuple[complex, int, int, np.ndarray]]:
    r"""
    Turns a square matrix into a series of hopping terms.

    Parameters
    ----------
    mat :
        The matrix to be converted.
    orbitals :
        Indices of the orbitals that make up the basis w.r.t. which the
        matrix is defined. By default (``orbitals=None``), the first
        ``len(mat)`` orbitals are used.
    R :
        Lattice vector for all the hopping terms.
    multiplier :
        Multiplicative constant for the hopping strength.
    """
    if orbitals is None:
        orbitals = list(range(len(mat)))
    hop = []
    for i, row in enumerate(mat):
        for j, x in enumerate(row):
            hop.append(
                (
                    multiplier * x,
                    orbitals[i],
                    orbitals[j],
                    np.array(R, dtype=int),
                )
            )
    return hop
