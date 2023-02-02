#
# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the :class:`.KdotpModel` class for k.p models.
"""

import typing as ty

import numpy as np
import numpy.typing as npt
import scipy.linalg as la

from fsc.hdf5_io import subscribe_hdf5, SimpleHDF5Mapping

__all__ = ("KdotpModel",)


@subscribe_hdf5("tbmodels.kdotp_model", check_on_load=False)
class KdotpModel(SimpleHDF5Mapping):
    """
    A class describing a k.p model.

    Parameters
    ----------
    taylor_coefficients:
        A mapping containing the taylor coefficients of the k.p model.
        The keys are tuples which describe the power of the k-vector
        components, and the values are the corresponding matrices.

        Example:
            (1, 0, 2): [[1, 0], [0, -1]]
            describes k_x * k_z**2 * sigma_z
    """

    HDF5_ATTRIBUTES = ["taylor_coefficients"]

    def __init__(  # pylint: disable=missing-function-docstring
        self, taylor_coefficients: ty.Mapping[ty.Tuple[int, ...], ty.Any]
    ) -> None:
        for mat in taylor_coefficients.values():
            if not np.allclose(mat, np.array(mat).T.conj()):
                raise ValueError(
                    f"The provided Taylor coefficient {mat} is not hermitian"
                )
        self.taylor_coefficients = {
            tuple(key): np.array(mat, dtype=complex)
            for key, mat in taylor_coefficients.items()
        }

    def hamilton(
        self, k: ty.Union[ty.Sequence[float], ty.Sequence[ty.Sequence[float]]]
    ) -> npt.NDArray[np.complex_]:
        """
        Calculates the Hamilton matrix for a given k-point or list of
        k-points.

        Parameters
        ----------
        k :
            The k-point at which the Hamiltonian is evaluated. If a list
            of k-points is given, the result will be the corresponding
            list of Hamiltonians.
        """
        k_array = np.array(k, ndmin=1)
        if k_array.ndim == 1:
            single_point = True
            k_array = k_array.reshape((1, -1))
        else:
            single_point = False

        ham = ty.cast(
            npt.NDArray[np.complex_],
            sum(
                np.prod(k_array**k_powers, axis=-1).reshape(-1, 1, 1)
                * mat[np.newaxis, :, :]
                for k_powers, mat in self.taylor_coefficients.items()
            ),
        )
        if single_point:
            return ty.cast(npt.NDArray[np.complex_], ham[0])
        return ham

    def eigenval(
        self, k: ty.Union[ty.Sequence[float], ty.Sequence[ty.Sequence[float]]]
    ) -> ty.Union[npt.NDArray[np.float_], ty.List[npt.NDArray[np.float_]]]:
        """
        Returns the eigenvalues at a given k point, or list of k-points.

        Parameters
        ----------
        k :
            The k-point at which the Hamiltonian is evaluated. If a list
            of k-points is given, a corresponding list of eigenvalue
            arrays is returned.
        """
        hamiltonians = self.hamilton(k)
        if hamiltonians.ndim == 3:
            return [la.eigvalsh(ham) for ham in hamiltonians]
        return ty.cast(npt.NDArray[np.float_], la.eigvalsh(hamiltonians))
