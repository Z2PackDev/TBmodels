import numpy as np
import scipy.linalg as la

from fsc.export import export
from fsc.hdf5_io import subscribe_hdf5, SimpleHDF5Mapping


@export
@subscribe_hdf5('tbmodels.model', check_on_load=False)
class KdotpModel(SimpleHDF5Mapping):
    HDF5_ATTRIBUTES = ['taylor_coefficients']

    def __init__(self, taylor_coefficients):
        self.taylor_coefficients = {
            tuple(key): np.array(mat, dtype=complex)
            for key, mat in taylor_coefficients.items()
        }

    def hamilton(self, k):
        return sum(sum(kval**p for kval, p in zip(k, pow)) * mat for pow, mat in self.taylor_coefficients.items())

    def eigenval(self, k):
        return la.eigvalsh(self.hamilton(k))
