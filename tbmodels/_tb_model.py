#
# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
# pylint: disable=too-many-lines,invalid-name
"""
Implements the Model class, which describes a tight-binding model.
"""

from __future__ import annotations

import re
import os
import copy
import time
import warnings
import itertools
import contextlib
import typing as ty
import collections as co

import h5py
import numpy as np
import scipy.linalg as la
from scipy.special import factorial
from fsc.export import export
from fsc.hdf5_io import subscribe_hdf5, HDF5Enabled

if ty.TYPE_CHECKING:
    # Replace with typing.Literal once Python 3.7 support is dropped.
    from typing_extensions import Literal
    import symmetry_representation  # pylint: disable=unused-import

from .kdotp import KdotpModel
from .exceptions import (
    TbmodelsException,
    ParseExceptionMarker,
    SymmetrizeExceptionMarker,
)
from . import _check_compatibility
from . import _sparse_matrix as sp

__all__ = ("Model",)

HoppingType = ty.Dict[ty.Tuple[int, ...], ty.Any]


@export
@subscribe_hdf5("tbmodels.model", check_on_load=False)
class Model(HDF5Enabled):
    """
    A class describing a tight-binding model. It contains methods for modifying the model, evaluating the Hamiltonian or eigenvalues at specific k-points, and writing to and from different file formats.

    Parameters
    ----------
    on_site :
        On-site energy of the states. This is equivalent to having a
        hopping within the same state and the same unit cell (diagonal
        terms of the R=(0, 0, 0) hopping matrix). The length of the list
        must be the same as the number of states.
    hop :
        Hopping matrices, as a dict containing the corresponding lattice
        vector R as a key.
    size :
        Number of states. Defaults to the size of the hopping matrices,
        if such are given.
    dim :
        Dimension of the tight-binding model. By default, the dimension
        is guessed from the other parameters if possible.
    occ :
        Number of occupied states.
    pos :
        Positions of the orbitals, in reduced coordinates. By default,
        all orbitals are set to be at the origin, i.e. at [0., 0., 0.].
    uc :
        Unit cell of the system. The unit cell vectors are given as rows
        in a ``dim`` x ``dim`` array
    contains_cc :
        Specifies whether the hopping matrices and on-site energies are
        given fully (``contains_cc=True``), or the complex conjugate
        should be added for each term to obtain the full model. The
        ``on_site`` parameter is not affected by this.
    cc_check_tolerance :
        Tolerance when checking if the complex conjugate values (if
        given) match.
    sparse :
        Specifies whether the hopping matrices should be saved in sparse
        format.
    """

    def __init__(
        self,
        *,
        on_site: ty.Optional[ty.Sequence[float]] = None,
        hop: ty.Optional[HoppingType] = None,
        size: ty.Optional[int] = None,
        dim: ty.Optional[int] = None,
        occ: ty.Optional[int] = None,
        pos: ty.Optional[ty.Sequence[ty.Sequence[float]]] = None,
        uc: ty.Optional[np.ndarray] = None,
        contains_cc: bool = True,
        cc_check_tolerance: float = 1e-12,
        sparse: bool = False,
    ):
        if hop is None:
            hop = dict()

        # ---- SPARSITY ----
        self._sparse: bool
        self._matrix_type: ty.Callable[..., ty.Any]
        self.set_sparse(sparse)

        # ---- SIZE ----
        self._init_size(size=size, on_site=on_site, hop=hop, pos=pos)

        # ---- DIMENSION ----
        self._init_dim(dim=dim, hop=hop, pos=pos, uc=uc)

        # ---- UNIT CELL ----
        self.uc = None if uc is None else np.array(uc)  # implicit copy

        # ---- HOPPING TERMS AND POSITIONS ----
        self._init_hop_pos(
            on_site=on_site,
            hop=hop,
            pos=pos,
            contains_cc=contains_cc,
            cc_check_tolerance=cc_check_tolerance,
        )

        # ---- CONSISTENCY CHECK FOR SIZE ----
        self._check_size_hop()

        # ---- CONSISTENCY CHECK FOR DIM ----
        self._check_dim()

        # ---- OCCUPATION NR ----
        self.occ = None if (occ is None) else int(occ)

    # ---------------- INIT HELPER FUNCTIONS --------------------------------#
    def _init_size(self, size, on_site, hop, pos):
        """
        Sets the size of the system (number of orbitals).
        """
        if size is not None:
            self.size = size
        elif on_site is not None:
            self.size = len(on_site)
        elif pos is not None:
            self.size = len(pos)
        elif hop:
            self.size = next(iter(hop.values())).shape[0]
        else:
            raise ValueError(
                "Empty hoppings dictionary supplied and no size, on-site energies or positions given. Cannot determine the size of the system."
            )

    def _init_dim(self, dim, hop, pos, uc):
        r"""
        Sets the system's dimensionality.
        """
        if dim is not None:
            self.dim = dim
        elif pos is not None:
            self.dim = len(pos[0])
        elif hop:
            self.dim = len(next(iter(hop.keys())))
        elif uc is not None:
            self.dim = len(uc[0])
        else:
            raise ValueError(
                "No dimension specified and no positions, hoppings, or unit cell are given. The dimensionality of the system cannot be determined."
            )

        self._zero_vec = tuple([0] * self.dim)

    def _init_hop_pos(self, on_site, hop, pos, contains_cc, cc_check_tolerance):
        """
        Sets the hopping terms and positions, mapping the positions to the UC (and changing the hoppings accordingly) if necessary.
        """
        # The double-constructor is needed to avoid a double-constructor in the sparse to-array
        # but still allow for the dtype argument.
        hop = {
            tuple(key): self._matrix_type(self._matrix_type(value), dtype=complex)
            for key, value in hop.items()
        }

        # positions
        if pos is None:
            self.pos = np.zeros((self.size, self.dim))
        elif len(pos) == self.size and all(len(p) == self.dim for p in pos):
            pos, hop = self._map_to_uc(pos, hop)
            self.pos = np.array(pos)  # implicit copy
        else:
            if len(pos) != self.size:
                raise ValueError(
                    "Invalid argument for 'pos': The number of positions must be the same as the size (number of orbitals) of the system."
                )
            raise ValueError(
                "Invalid argument for 'pos': The length of each position must be the same as the dimensionality of the system."
            )

        if contains_cc:
            hop = self._reduce_hop(hop, cc_check_tolerance=cc_check_tolerance)
        else:
            hop = self._map_hop_positive_R(hop)
        # use partial instead of lambda to allow for pickling
        self.hop = co.defaultdict(self._empty_matrix)
        for R, h_mat in hop.items():
            if not np.any(h_mat):
                continue
            self.hop[R] = self._matrix_type(h_mat)
        # add on-site terms
        if on_site is not None:
            if len(on_site) != self.size:
                raise ValueError(
                    "The number of on-site energies {} does not match the size of the system {}".format(
                        len(on_site), self.size
                    )
                )
            self.hop[self._zero_vec] += 0.5 * self._matrix_type(np.diag(on_site))

    # helpers for _init_hop_pos
    def _map_to_uc(self, pos, hop):
        """
        hoppings in csr format
        """
        uc_offsets = [np.array(np.floor(p), dtype=int) for p in pos]
        # ---- common case: already mapped into the UC ----
        if all([all(o == 0 for o in offset) for offset in uc_offsets]):
            return pos, hop

        # ---- uncommon case: handle mapping ----
        new_pos = [np.array(p) % 1 for p in pos]
        new_hop = co.defaultdict(
            lambda: np.zeros((self.size, self.size), dtype=complex)
        )
        for R, hop_mat in hop.items():
            hop_mat = np.array(hop_mat)
            for i0, row in enumerate(hop_mat):
                for i1, t in enumerate(row):
                    if t != 0:
                        R_new = tuple(
                            np.array(R, dtype=int) + uc_offsets[i1] - uc_offsets[i0]
                        )
                        new_hop[R_new][i0][i1] += t
        new_hop = {key: self._matrix_type(value) for key, value in new_hop.items()}
        return new_pos, new_hop

    @staticmethod
    def _reduce_hop(hop, cc_check_tolerance):
        """
        Reduce the full hoppings representation (with cc) to the reduced one (without cc, zero-terms halved).
        """
        # Consistency checks
        failed_R = []
        res = dict()
        for R, mat in hop.items():
            equiv_mat = hop.get(tuple(-x for x in R), np.zeros(mat.shape)).T.conjugate()
            diff_norm = la.norm(mat - equiv_mat)
            if diff_norm > cc_check_tolerance:
                failed_R.append((R, diff_norm))

            avg_mat = (mat + equiv_mat) / 2

            try:
                if R[np.nonzero(R)[0][0]] > 0:
                    res[R] = avg_mat
            # Case R = 0
            except IndexError:
                res[R] = avg_mat / 2

        if failed_R:
            raise ValueError(
                "The provided hoppings do not correspond to a hermitian Hamiltonian. hoppings[-R] = hoppings[R].H is not fulfilled for the following values:\n"
                + "\n".join(
                    f"R={R}, delta_norm={diff_norm}"
                    for R, diff_norm in sorted(failed_R, key=lambda val: -val[1])
                )
            )

        return res

    def _map_hop_positive_R(self, hop: HoppingType) -> HoppingType:
        """
        Maps hoppings with a negative first non-zero index in R to their positive counterpart.
        """
        new_hop: HoppingType = co.defaultdict(self._empty_matrix)
        for R, mat in hop.items():
            try:
                if R[np.nonzero(R)[0][0]] > 0:
                    new_hop[R] += mat
                else:
                    minus_R = tuple(-x for x in R)
                    new_hop[minus_R] += mat.transpose().conjugate()
            except IndexError:
                # make sure the zero term is also hermitian
                # This only really needed s.t. the representation is unique.
                # The Hamiltonian is anyway made hermitian later.
                new_hop[R] += 0.5 * mat + 0.5 * mat.conjugate().transpose()
        return new_hop

    # end helpers for _init_hop_pos

    def _check_size_hop(self):
        """
        Consistency check for the size of the hopping matrices.
        """
        for h_mat in self.hop.values():
            if not h_mat.shape == (self.size, self.size):
                raise ValueError(
                    "Hopping matrix of shape {0} found, should be ({1},{1}).".format(
                        h_mat.shape, self.size
                    )
                )

    def _check_dim(self):
        """Consistency check for the dimension of the hoppings and unit cell. The position is checked in _init_hop_pos"""
        for key in self.hop.keys():
            if len(key) != self.dim:
                raise ValueError(
                    "The length of R = {} does not match the dimensionality of the system ({})".format(
                        key, self.dim
                    )
                )
        if self.uc is not None:
            if self.uc.shape != (self.dim, self.dim):
                raise ValueError(
                    "Inconsistend dimension of the unit cell: {}, does not match the dimensionality of the system ({})".format(
                        self.uc.shape, self.dim
                    )
                )

    # ---------------- CONSTRUCTORS / (DE)SERIALIZATION ----------------#
    @classmethod
    def from_hop_list(
        cls,
        *,
        hop_list: ty.Iterable[ty.Tuple[complex, int, int, ty.Tuple[int, ...]]] = (),
        size: ty.Optional[int] = None,
        **kwargs,
    ) -> Model:
        """
        Create a :class:`.Model` from a list of hopping terms.

        Parameters
        ----------
        hop_list :
            List of hopping terms. Each hopping term has the form
            [t, orbital_1, orbital_2, R], where

                * ``t``: strength of the hopping
                * ``orbital_1``: index of the first involved orbital
                * ``orbital_2``: index of the second involved orbital
                * ``R``: lattice vector of the unit cell containing the second orbital.
        size :
            Number of states. Defaults to the length of the on-site energies given, if such are given.
        kwargs :
            Any :class:`.Model` keyword arguments.
        """
        if size is None:
            try:
                size = len(kwargs["on_site"])
            except KeyError as exc:
                raise ValueError(
                    "No on-site energies and no size given. The size of the system cannot be determined."
                ) from exc

        class _hop:
            """
            POD for hoppings
            """

            def __init__(self):
                self.data = []
                self.row_idx = []
                self.col_idx = []

            def append(self, data, row_idx, col_idx):
                self.data.append(data)
                self.row_idx.append(row_idx)
                self.col_idx.append(col_idx)

        # create data, row_idx, col_idx for setting up the CSR matrices
        hop_list_dict: ty.Mapping[ty.Tuple[int, ...], _hop] = co.defaultdict(_hop)
        R: ty.Tuple[int, ...]
        for t, i, j, R in hop_list:
            R_vec = tuple(R)
            hop_list_dict[R_vec].append(t, i, j)

        # creating CSR matrices
        hop_dict = dict()
        for key, val in hop_list_dict.items():
            hop_dict[key] = sp.csr(
                (val.data, (val.row_idx, val.col_idx)),
                dtype=complex,
                shape=(size, size),
            )

        return cls(size=size, hop=hop_dict, **kwargs)

    @staticmethod
    def _read_hr(iterator, ignore_orbital_order=False):
        r"""
        read the number of wannier functions and the hopping entries
        from *hr.dat and converts them into the right format
        """
        next(iterator)  # skip first line
        num_wann = int(next(iterator))
        nrpts = int(next(iterator))

        # get degeneracy points
        deg_pts = []
        # order in zip important because else the next data element is consumed
        for _, line in zip(range(int(np.ceil(nrpts / 15))), iterator):
            deg_pts.extend(int(x) for x in line.split())
        assert len(deg_pts) == nrpts

        num_wann_square = num_wann ** 2

        def to_entry(line, i):
            """Turns a line (string) into a hop_list entry"""
            entry = line.split()
            orbital_a = int(entry[3]) - 1
            orbital_b = int(entry[4]) - 1
            # test consistency of orbital numbers
            if not ignore_orbital_order:
                if not (orbital_a == i % num_wann) and (
                    orbital_b == (i % num_wann_square) // num_wann
                ):
                    raise ValueError(f"Inconsistent orbital numbers in line '{line}'")
            return [
                (float(entry[5]) + 1j * float(entry[6]))
                / (deg_pts[i // num_wann_square]),
                orbital_a,
                orbital_b,
                [int(x) for x in entry[:3]],
            ]

        # skip random empty lines
        lines_nonempty = (l for l in iterator if l.strip())
        hop_list = (to_entry(line, i) for i, line in enumerate(lines_nonempty))

        return num_wann, hop_list

    def to_hr_file(self, hr_file: str) -> None:
        """
        Writes to a file, using Wannier90's ``*_hr.dat`` format.

        Parameters
        ----------
        hr_file :
            Path of the output file


        .. note :: The ``*_hr.dat`` format does not contain information
            about the position of the atoms or the shape of the unit
            cell. Consequently, this information is lost when saving the
            model in this format.

        .. warning :: The ``*_hr.dat`` format does not preserve the full
            precision of the hopping strengths. This could lead to
            numerical errors.
        """
        with open(hr_file, "w") as f:
            f.write(self.to_hr())

    def to_hr(self) -> str:
        """
        Returns a string containing the model in Wannier90's
        ``*_hr.dat`` format.

        .. note :: The ``*_hr.dat`` format does not contain information about the position of the atoms or the shape of the unit cell. Consequently, this information is lost when saving the model in this format.

        .. warning :: The ``*_hr.dat`` format does not preserve the full precision of the hopping strengths. This could lead to numerical errors.
        """
        lines = []
        tagline = " created by the TBmodels package    " + time.strftime(
            "%a, %d %b %Y %H:%M:%S %Z"
        )
        lines.append(tagline)
        lines.append(f"{self.size:>12}")
        num_g = len(self.hop.keys()) * 2 - 1
        if num_g <= 0:
            raise ValueError("Cannot print empty model to hr format.")
        lines.append(f"{num_g:>12}")
        tmp = ""
        for i in range(num_g):
            if tmp != "" and i % 15 == 0:
                lines.append(tmp)
                tmp = ""
            tmp += "    1"
        lines.append(tmp)

        # negative
        for R in reversed(sorted(self.hop.keys())):
            if R != self._zero_vec:
                minus_R = tuple(-x for x in R)
                lines.extend(
                    self._mat_to_hr(minus_R, self.hop[R].conjugate().transpose())
                )
        # zero
        if self._zero_vec in self.hop.keys():
            lines.extend(
                self._mat_to_hr(
                    self._zero_vec,
                    self.hop[self._zero_vec]
                    + self.hop[self._zero_vec].conjugate().transpose(),
                )
            )
        # positive
        for R in sorted(self.hop.keys()):
            if R != self._zero_vec:
                lines.extend(self._mat_to_hr(R, self.hop[R]))

        return "\n".join(lines)

    @staticmethod
    def _mat_to_hr(R, mat):
        """
        Creates the ``*_hr.dat`` string for a single hopping matrix.
        """
        lines = []
        mat = np.array(mat).T  # to be consistent with W90's ordering
        for j, column in enumerate(mat):
            for i, t in enumerate(column):
                lines.append(
                    "{0[0]:>5}{0[1]:>5}{0[2]:>5}{1:>5}{2:>5}{3.real:>22.14f}{3.imag:>22.14f}".format(
                        R, i + 1, j + 1, t
                    )
                )
        return lines

    @classmethod
    def from_wannier_folder(
        cls, folder: str = ".", prefix: str = "wannier", **kwargs
    ) -> Model:
        """
        Create a :class:`.Model` instance from Wannier90 output files,
        given the folder containing the files and file prefix.

        Parameters
        ----------
        folder :
            Directory containing the Wannier90 output files.
        prefix :
            Prefix of the Wannier90 output files.
        kwargs :
            Keyword arguments passed to :meth:`.from_wannier_files`. If
            input files are explicitly given, they take precedence over
            those found in the ``folder``.
        """
        common_path = os.path.join(folder, prefix)
        input_files = dict()
        input_files["hr_file"] = common_path + "_hr.dat"

        for key, suffix in [
            ("win_file", ".win"),
            ("wsvec_file", "_wsvec.dat"),
            ("xyz_file", "_centres.xyz"),
        ]:
            filename = common_path + suffix
            if os.path.isfile(filename):
                input_files[key] = filename

        return cls.from_wannier_files(**co.ChainMap(kwargs, input_files))

    @classmethod  # noqa: MC0001
    def from_wannier_files(  # pylint: disable=too-many-locals
        cls,
        *,
        hr_file: str,
        wsvec_file: ty.Optional[str] = None,
        xyz_file: ty.Optional[str] = None,
        win_file: ty.Optional[str] = None,
        h_cutoff: float = 0.0,
        ignore_orbital_order: bool = False,
        pos_kind: str = "wannier",
        distance_ratio_threshold: float = 3.0,
        **kwargs,
    ) -> Model:
        """
        Create a :class:`.Model` instance from Wannier90 output files.

        Parameters
        ----------
        hr_file :
            Path of the ``*_hr.dat`` file. Together with the
            ``*_wsvec.dat`` file, this determines the hopping terms.
        wsvec_file :
            Path of the ``*_wsvec.dat`` file. This file determines the
            remapping of hopping terms when ``use_ws_distance`` is used
            in the Wannier90 calculation.
        xyz_file :
            Path of the ``*_centres.xyz`` file. This file is used to
            determine the positions of the orbitals, from the Wannier
            centers given by Wannier90.
        win_file :
            Path of the ``*.win`` file. This file is used to determine
            the unit cell.
        h_cutoff :
            Cutoff value for the hopping strength. Hoppings with a
            smaller absolute value are ignored.
        ignore_orbital_order :
            Do not throw an error when the order of orbitals does not
            match what is expected from the Wannier90 output.
        pos_kind :
            Determines how positions are assinged to orbitals. Valid
            options are `wannier` (use Wannier centres) or
            `nearest_atom` (map to nearest atomic position).
        distance_ratio_threshold :
            [Applies only for pos_kind='nearest_atom']
            The minimum ratio between the second-nearest and nearest
            atom below which an error will be raised.
        kwargs :
            :class:`.Model` keyword arguments.
        """

        if win_file is not None:
            if "uc" in kwargs:
                raise ValueError(
                    "Ambiguous unit cell: It can be given either via 'uc' or the 'win_file' keywords, but not both."
                )
            with open(win_file) as f:
                kwargs["uc"] = cls._read_win(f)["unit_cell_cart"]

        if xyz_file is not None:
            if "pos" in kwargs:
                raise ValueError(
                    "Ambiguous orbital positions: The positions can be given either via the 'pos' or the 'xyz_file' keywords, but not both."
                )
            if "uc" not in kwargs:
                raise ValueError(
                    "Positions cannot be read from .xyz file without unit cell given: Transformation from cartesian to reduced coordinates not possible. Specify the unit cell using one of the keywords 'uc' or 'win_file'."
                )
            with open(xyz_file) as f:
                wannier_pos_list_cartesian, atom_list_cartesian = cls._read_xyz(f)
                wannier_pos_cartesian = np.array(wannier_pos_list_cartesian)
                atom_pos_cartesian = np.array([a.pos for a in atom_list_cartesian])
                if pos_kind == "wannier":
                    pos_cartesian: ty.Union[
                        ty.List[np.ndarray], np.ndarray
                    ] = wannier_pos_cartesian
                elif pos_kind == "nearest_atom":
                    if distance_ratio_threshold < 1:
                        raise ValueError(
                            "Invalid value for 'distance_ratio_threshold': must be >= 1."
                        )
                    pos_cartesian = ty.cast(ty.List[np.ndarray], [])
                    for p in wannier_pos_cartesian:
                        p_reduced = la.solve(kwargs["uc"].T, np.array(p).T).T
                        T_base = np.floor(p_reduced)
                        all_atom_pos = np.array(
                            [
                                kwargs["uc"].T @ (T_base + T_shift) + atom_pos
                                for atom_pos in atom_pos_cartesian
                                for T_shift in itertools.product([-1, 0, 1], repeat=3)
                            ]
                        )
                        distances = la.norm(p - all_atom_pos, axis=-1)
                        idx = np.argpartition(distances, 2)[:2]
                        nearest, second_nearest = distances[idx]
                        if second_nearest / nearest < distance_ratio_threshold:
                            raise TbmodelsException(
                                f"The ratio ({second_nearest / nearest:.3f}) between "
                                f"the nearest ({nearest:.3f}) and second-nearest "
                                f"({second_nearest:.3f}) atomic position is less than "
                                f"'distance_ratio_threshold' ({distance_ratio_threshold}).",
                                exception_marker=ParseExceptionMarker.AMBIGUOUS_NEAREST_ATOM_POSITIONS,
                            )
                        pos_cartesian.append(all_atom_pos[idx[0]])
                else:
                    raise ValueError(
                        "Invalid value '{}' for 'pos_kind', must be 'wannier' or 'nearest_atom'".format(
                            pos_kind
                        )
                    )
                kwargs["pos"] = la.solve(kwargs["uc"].T, np.array(pos_cartesian).T).T

        with open(hr_file) as f:
            num_wann, hop_entries = cls._read_hr(
                f, ignore_orbital_order=ignore_orbital_order
            )
            hop_entries = (hop for hop in hop_entries if abs(hop[0]) > h_cutoff)

            if wsvec_file is not None:
                with open(wsvec_file) as f:
                    wsvec_generator = cls._async_parse(
                        cls._read_wsvec(f), chunksize=num_wann
                    )

                    def remap_hoppings(hop_entries):
                        for t, orbital_1, orbital_2, R in hop_entries:
                            # Step _async_parse to where it accepts
                            # a new key.
                            # The _async_parse does not raise StopIteration
                            next(  # pylint: disable=stop-iteration-return
                                wsvec_generator
                            )
                            T_list = wsvec_generator.send(
                                (orbital_1, orbital_2, tuple(R))
                            )
                            N = len(T_list)
                            for T in T_list:
                                # not using numpy here increases performance
                                yield (
                                    t / N,
                                    orbital_1,
                                    orbital_2,
                                    tuple(r + t for r, t in zip(R, T)),
                                )

                    hop_entries = remap_hoppings(hop_entries)
                    return cls.from_hop_list(
                        size=num_wann, hop_list=hop_entries, **kwargs
                    )

            return cls.from_hop_list(size=num_wann, hop_list=hop_entries, **kwargs)

    @staticmethod
    def _async_parse(iterator, chunksize=1):
        """
        Helper function to get values from a (key, value) iterator
        out of order without having to exhaust the iterator from the start.
        The desired key needs to be sent to this generator, and it
        will go through the `iterator` until that key is found. Pairs
        for which the key has not yet been requested are stored in a
        temporary dictionary.

        Note that this generator never raises StopIteration, it can
        only exit with KeyError.
        """
        mapping = dict()
        stopped = False
        while True:
            # get the desired key
            key = yield
            while True:
                try:
                    # key found
                    yield mapping.pop(key)
                    break
                except KeyError as e:
                    if stopped:
                        # avoid infinte loop in true KeyError
                        raise e
                    for _ in range(chunksize):
                        try:
                            # parse new data
                            newkey, newval = next(iterator)
                            mapping[newkey] = newval
                        except StopIteration:
                            stopped = True
                            break

    @staticmethod
    def _read_wsvec(iterator):
        """
        Generator that parses the content of the *_wsvec.dat file.
        """
        # skip comment line
        try:
            next(iterator)
        except StopIteration as exc:
            raise TbmodelsException(
                "The 'wsvec' iterator is empty.",
                exception_marker=ParseExceptionMarker.INCOMPLETE_WSVEC_FILE,
            ) from exc
        for first_line in iterator:
            *R, o1, o2 = (int(x) for x in first_line.split())
            # in our convention, orbital indices start at 0.
            key = (o1 - 1, o2 - 1, tuple(R))
            try:
                N = int(next(iterator))
                val = [tuple(int(x) for x in next(iterator).split()) for _ in range(N)]
            except StopIteration as exc:
                raise TbmodelsException(
                    "Incomplete wsvec iterator.",
                    exception_marker=ParseExceptionMarker.INCOMPLETE_WSVEC_FILE,
                ) from exc
            yield key, val

    @staticmethod
    def _read_xyz(iterator):
        """Reads the content of a .xyz file"""
        # This functionality exists within pymatgen, so it might make sense
        # to use that if we anyway want pymatgen as a dependency.
        N = int(next(iterator))
        next(iterator)  # skip comment line
        wannier_centres = []
        atom_positions = []
        AtomPosition = co.namedtuple("AtomPosition", ["kind", "pos"])
        for l in iterator:
            kind, *pos = l.split()
            pos = tuple(float(x) for x in pos)
            if kind == "X":
                wannier_centres.append(pos)
            else:
                atom_positions.append(AtomPosition(kind=kind, pos=pos))
        assert len(wannier_centres) + len(atom_positions) == N
        return wannier_centres, atom_positions

    @staticmethod
    def _read_win(iterator):
        """
        Takes an iterator representing the Wannier90 .win file lines,
        and returns a mapping of its content.
        """
        lines = (l.split("!")[0] for l in iterator)
        lines = (l.strip() for l in lines)
        lines = (l for l in lines if l)
        lines = (l.lower() for l in lines)

        split_token = re.compile("[\t :=]+")

        mapping = {}
        for line in lines:
            if line.startswith("begin"):
                key = split_token.split(line[5:].strip(" :="), 1)[0]
                val = []
                while True:
                    line = next(lines)
                    if line.startswith("end"):
                        end_key = split_token.split(line[3:].strip(" :="), 1)[0]
                        assert end_key == key
                        break
                    val.append(line)
                mapping[key] = val
            else:
                key, val = split_token.split(line, 1)
                mapping[key] = val

        # here we can continue parsing the individual keys as needed
        if "length_unit" in mapping:
            length_unit = mapping["length_unit"].strip().lower()
        else:
            length_unit = "ang"
        mapping["length_unit"] = length_unit

        if "unit_cell_cart" in mapping:
            uc_input = mapping["unit_cell_cart"]
            # handle the case when the unit is explicitly given
            if len(uc_input) == 4:
                unit, *uc_input = uc_input
                # unit = unit[0]
            else:
                unit = length_unit
            val = [[float(x) for x in split_token.split(line)] for line in uc_input]
            val = np.array(val).reshape(3, 3)
            if unit == "bohr":
                val *= 0.52917721092
            mapping["unit_cell_cart"] = val

        return mapping

    def to_kwant_lattice(self):
        """
        Returns a kwant lattice corresponding to the current model. Orbitals with the same position are grouped into the same Monoatomic sublattice.

        .. note :: The TBmodels - Kwant interface is experimental. Use it with caution.
        """
        import kwant  # pylint: disable=import-outside-toplevel

        sublattices = self._get_sublattices()
        uc = self.uc if self.uc is not None else np.eye(self.dim)
        # get sublattice positions in cartesian coordinates
        pos_abs = np.dot(np.array([sl.pos for sl in sublattices]), uc)
        return kwant.lattice.general(prim_vecs=uc, basis=pos_abs)

    def add_hoppings_kwant(self, kwant_sys):
        """
        Sets the on-site energies and hopping terms for an existing kwant system to those of the :class:`.Model`.

        .. note :: The TBmodels - Kwant interface is experimental. Use it with caution.
        """
        import kwant  # pylint: disable=import-outside-toplevel

        sublattices = self._get_sublattices()
        kwant_sublattices = self.to_kwant_lattice().sublattices

        # handle R = 0 case (on-site)
        on_site_mat = copy.deepcopy(self._array_cast(self.hop[self._zero_vec]))
        on_site_mat += on_site_mat.conjugate().transpose()
        # R = 0 terms within a sublattice (on-site)
        for site in kwant_sys.sites():
            for i, latt in enumerate(kwant_sublattices):
                if site.family == latt:
                    indices = sublattices[i].indices
                    kwant_sys[site] = on_site_mat[np.ix_(indices, indices)]
                    break
            # site doesn't belong to any sublattice
            else:
                # TODO: check if there is a legitimate use case which triggers this
                raise ValueError(f"Site {site} did not match any sublattice.")

        # R = 0 terms between different sublattices
        for i, s1 in enumerate(sublattices):
            for j, s2 in enumerate(sublattices):
                if i == j:
                    # handled above
                    continue
                kwant_sys[
                    kwant.builder.HoppingKind(
                        self._zero_vec,
                        kwant_sublattices[i],
                        kwant_sublattices[j],
                    )
                ] = on_site_mat[np.ix_(s1.indices, s2.indices)]

        # R != 0 terms
        for R, mat in self.hop.items():
            # special case R = 0 handled already
            if R == self._zero_vec:
                continue
            mat = self._array_cast(mat)
            minus_R = tuple(-np.array(R))
            for i, s1 in enumerate(sublattices):
                for j, s2 in enumerate(sublattices):
                    sub_matrix = mat[np.ix_(s1.indices, s2.indices)]
                    # TODO: check "signs"
                    kwant_sys[
                        kwant.builder.HoppingKind(
                            minus_R, kwant_sublattices[i], kwant_sublattices[j]
                        )
                    ] = sub_matrix
                    kwant_sys[
                        kwant.builder.HoppingKind(
                            R, kwant_sublattices[j], kwant_sublattices[i]
                        )
                    ] = np.transpose(np.conj(sub_matrix))
        return kwant_sys

    def _get_sublattices(self):
        """
        Helper function to group indices of orbitals which have the same
        position into sublattices.
        """
        Sublattice = co.namedtuple("Sublattice", ["pos", "indices"])
        sublattices = []
        for i, p_orb in enumerate(self.pos):
            # try to match an existing sublattice
            for sub_pos, sub_indices in sublattices:
                if np.isclose(p_orb, sub_pos, rtol=0).all():
                    sub_indices.append(i)
                    break
            # create new sublattice
            else:
                sublattices.append(Sublattice(pos=p_orb, indices=[i]))
        return sublattices

    def construct_kdotp(self, k: ty.Sequence[float], order: int):
        """
        Construct a k.p model around a given k-point. This is done by explicitly
        evaluating the derivatives which make up the Taylor expansion of the k.p
        models.

        This method can currently only construct models using
        `convention 2  <http://www.physics.rutgers.edu/pythtb/_downloads/pythtb-formalism.pdf>`_
        for the Hamiltonian.

        Parameters
        ----------
        k :
            The k-point around which the k.p model is constructed.
        order :
            The order (sum of powers) to which the Taylor expansion is
            performed.
        """
        taylor_coefficients = dict()
        if order < 0:
            raise ValueError("The order for the k.p model must be positive.")
        k_powers: ty.Tuple[int, ...]
        for k_powers in itertools.product(range(order + 1), repeat=self.dim):
            curr_order = sum(k_powers)
            if curr_order > order:
                continue
            taylor_coefficients[k_powers] = (
                (2j * np.pi) ** curr_order / np.prod(factorial(k_powers, exact=True))
            ) * sum(
                (
                    np.prod(np.array(R) ** np.array(k_powers))
                    * np.exp(2j * np.pi * np.dot(k, R))
                    * self._array_cast(mat)
                    + np.prod((-np.array(R)) ** np.array(k_powers))
                    * np.exp(-2j * np.pi * np.dot(k, R))
                    * self._array_cast(mat).T.conj()
                    for R, mat in self.hop.items()
                ),
                np.zeros((self.size, self.size), dtype=complex),
            )
        return KdotpModel(taylor_coefficients=taylor_coefficients)

    @classmethod
    def from_hdf5_file(  # pylint: disable=arguments-differ
        cls, hdf5_file: str, **kwargs
    ) -> Model:
        """
        Returns a :class:`.Model` instance read from a file in HDF5
        format.

        Parameters
        ----------
        hdf5_file :
            Path of the input file.
        kwargs :
            :class:`.Model` keyword arguments. Explicitly specified
            keywords take precedence over those given in the HDF5 file.
        """
        with h5py.File(hdf5_file, "r") as hdf_handle:
            if "type_tag" not in hdf_handle:
                warnings.warn(
                    f"The loaded file '{hdf5_file}' is stored in an outdated "
                    "format. Consider loading and storing the file to update it.",
                    DeprecationWarning,
                )
            return cls.from_hdf5(hdf_handle, **kwargs)

    @classmethod
    def from_hdf5(  # pylint: disable=arguments-differ
        cls, hdf5_handle, **kwargs
    ) -> Model:
        # For compatibility with a development version which created a top-level
        # 'tb_model' attribute.
        try:
            tb_model_group = hdf5_handle["tb_model"]
        except KeyError:
            tb_model_group = hdf5_handle
        new_kwargs: ty.Dict[str, ty.Any] = {}
        new_kwargs["hop"] = {}

        for key in ["uc", "occ", "size", "dim", "pos", "sparse"]:
            if key in tb_model_group:
                new_kwargs[key] = tb_model_group[key][()]

        if "hop" not in kwargs:
            for group in tb_model_group["hop"].values():
                R = tuple(group["R"])
                if new_kwargs["sparse"]:
                    new_kwargs["hop"][R] = sp.csr(
                        (group["data"], group["indices"], group["indptr"]),
                        shape=group["shape"],
                    )
                else:
                    new_kwargs["hop"][R] = np.array(group["mat"])
            new_kwargs["contains_cc"] = False
        return cls(**co.ChainMap(kwargs, new_kwargs))

    def to_hdf5(self, hdf5_handle):
        if self.uc is not None:
            hdf5_handle["uc"] = self.uc
        if self.occ is not None:
            hdf5_handle["occ"] = self.occ
        hdf5_handle["size"] = self.size
        hdf5_handle["dim"] = self.dim
        hdf5_handle["pos"] = self.pos
        hdf5_handle["sparse"] = self._sparse
        hop = hdf5_handle.create_group("hop")
        for i, (R, mat) in enumerate(self.hop.items()):
            group = hop.create_group(str(i))
            group["R"] = R
            if self._sparse:
                group["data"] = mat.data
                group["indices"] = mat.indices
                group["indptr"] = mat.indptr
                group["shape"] = mat.shape
            else:
                group["mat"] = mat

    def __repr__(self):
        # Note: this is affected by issue #76
        return " ".join(
            f"tbmodels.Model(hop=<{len(self.hop)} matrices>, pos=<{len(self.pos)} values>, uc={self.uc!r}, occ={self.occ}, contains_cc=False)".replace(
                "\n", " "
            )
            .replace("array", "np.array")
            .split()
        )

    # ---------------- BASIC FUNCTIONALITY ----------------------------------#
    @property
    def reciprocal_lattice(self):
        """An array containing the reciprocal lattice vectors as rows."""
        return None if self.uc is None else 2 * np.pi * la.inv(self.uc).T

    def hamilton(
        self,
        k: ty.Union[ty.Sequence[float], ty.Sequence[ty.Sequence[float]]],
        convention: int = 2,
    ) -> np.ndarray:
        """
        Calculates the Hamilton matrix for a given k-point or list of
        k-points.

        Parameters
        ----------
        k :
            The k-point at which the Hamiltonian is evaluated. If a list
            of k-points is given, the result will be the corresponding
            list of Hamiltonians.
        convention :
            Choice of convention to calculate the Hamilton matrix. See
            explanation in `the PythTB documentation
            <http://www.physics.rutgers.edu/pythtb/_downloads/pythtb-formalism.pdf>`_ .
            Valid choices are 1 or 2.
        """
        if convention not in [1, 2]:
            raise ValueError(
                "Invalid value '{}' for 'convention': must be either '1' or '2'".format(
                    convention
                )
            )
        k_array = np.array(k, ndmin=1)
        if k_array.ndim == 1:
            single_point = True
            k_array = k_array.reshape((1, -1))
        else:
            single_point = False
        H = np.zeros((k_array.shape[0], self.size, self.size), dtype=complex)
        tmp_array = np.empty_like(H)
        for R, hop in self.hop.items():
            # When the hopping matrices are very large, allocating new
            # arrays for the result of this multiplication (which is
            # of size len(k_array) * self.size**2) becomes expensive.
            # To avoid this, we reuse the same temporary array - even
            # if this is _slightly_ slower for single k-point calculations.
            np.multiply(
                np.exp(2j * np.pi * np.dot(k_array, R)).reshape((-1, 1, 1)),
                self._array_cast(hop)[np.newaxis, :, :],
                out=tmp_array,
            )
            H += tmp_array
        H += H.conjugate().transpose((0, 2, 1))
        if convention == 1:
            pos_exponential = np.array(
                [[np.exp(2j * np.pi * np.dot(k_array, p)) for p in self.pos]]
            ).transpose((2, 0, 1))
            H = pos_exponential.conjugate().transpose((0, 2, 1)) * H * pos_exponential

        if single_point:
            return ty.cast(np.ndarray, H[0])
        return H

    def eigenval(
        self, k: ty.Union[ty.Sequence[float], ty.Sequence[ty.Sequence[float]]]
    ) -> ty.Union[np.ndarray, ty.List[np.ndarray]]:
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
        return ty.cast(np.ndarray, la.eigvalsh(hamiltonians))

    # -------------------MODIFYING THE MODEL ----------------------------#
    def add_hop(
        self,
        overlap: complex,
        orbital_1: int,
        orbital_2: int,
        R: ty.Sequence[int],
    ):
        r"""
        Adds a hopping term with a given overlap (hopping strength) from
        ``orbital_2`` (:math:`o_2`), which lies in the unit cell pointed
        to by ``R``, to ``orbital_1`` (:math:`o_1`) which is in the home
        unit cell. In other words, ``overlap`` is the matrix element
        :math:`\mathcal{H}_{o_1,o_2}(\mathbf{R}) = \langle o_1, \mathbf{0} | \mathcal{H} | o_2, \mathbf{R} \rangle`.

        The complex conjugate of the hopping is added automatically.
        That is, the matrix element
        :math:`\langle o_2, \mathbf{R} | \mathcal{H} | o_1, \mathbf{0} \rangle`
        does not have to be added manually.

        .. note::
            This means that adding a hopping of overlap :math:`\epsilon`
            between an orbital and itself in the home unit cell
            increases the orbitals on-site energy by :math:`2 \epsilon`.


        Parameters
        ----------
        overlap :
            Strength of the hopping term (in energy units).
        orbital_1 :
            Index of the first orbital.
        orbital_2 :
            Index of the second orbital.
        R :
            Lattice vector pointing to the unit cell where ``orbital_2``
            lies.


        .. warning::
            The positions given in the constructor of :class:`Model`
            are automatically mapped into the home unit cell. This has
            to be taken into account when determining ``R``.

        """
        R = tuple(R)
        if len(R) != self.dim:
            raise ValueError(
                "Dimension of R ({}) does not match the model dimension ({})".format(
                    len(R), self.dim
                )
            )

        mat = np.zeros((self.size, self.size), dtype=complex)
        nonzero_idx = np.nonzero(R)[0]
        if nonzero_idx.size == 0:
            mat[orbital_1, orbital_2] += overlap / 2.0
            mat[orbital_2, orbital_1] += overlap.conjugate() / 2.0
        elif R[nonzero_idx[0]] > 0:
            mat[orbital_1, orbital_2] += overlap
        else:
            R = tuple(-x for x in R)
            mat[orbital_2, orbital_1] += overlap.conjugate()
        self.hop[R] += self._matrix_type(mat)

    def add_on_site(self, on_site: ty.Sequence[float]):
        """
        Adds on-site energy to the orbitals. This adds to the existing
        on-site energy, and does not erase it.

        Parameters
        ----------
        on_site :
            On-site energies. This must be a sequence of real numbers, of the same length as the number of orbitals
        """
        if self.size != len(on_site):
            raise ValueError(
                "The number of on-site energy terms should be {}, but is {}.".format(
                    self.size, len(on_site)
                )
            )
        for orbital, energy in enumerate(on_site):
            self.add_hop(energy / 2.0, orbital, orbital, self._zero_vec)

    def remove_small_hop(self, cutoff: float) -> None:
        """Remove hoppings which are smaller than the given cutoff value.

        Parameters
        ----------
        cutoff :
            Cutoff value below which a hopping is removed.
        """
        # Cast to list because the dictionary is modified in the loop.
        for R, hop_mat in list(self.hop.items()):
            curr_cutoff = cutoff
            if R == self._zero_vec:
                curr_cutoff /= 2

            hop_mat_arr = self._array_cast(hop_mat)
            hop_mat_arr[np.abs(hop_mat_arr) < curr_cutoff] = 0.0

            if not np.any(hop_mat_arr):
                del self.hop[R]
            else:
                self.hop[R] = self._matrix_type(hop_mat_arr)

    def remove_long_range_hop(self, *, cutoff_distance_cartesian: float) -> None:
        """Remove hoppings whose range is longer than the given cutoff.

        Parameters
        ----------
        cutoff_distance_cartesian :
            Cartesian distance between the two orbitals above which the
            hoppings are removed.
        """
        if self.uc is None or self.pos is None:
            raise ValueError(
                "Both the unit cell and positions must be specified to "
                "determine the cartesian distance between orbitals."
            )
        pos_cart = (self.uc.T @ self.pos.T).T
        pos_offset_cart = pos_cart.reshape((1, -1, self.dim)) - pos_cart.reshape(
            (-1, 1, self.dim)
        )

        # Cast to list because the dictionary is modified in the loop.
        for R, hop_mat in list(self.hop.items()):
            R_cart = self.uc.T @ R
            distances = la.norm(R_cart + pos_offset_cart, axis=-1)

            hop_mat_arr = self._array_cast(hop_mat)
            hop_mat_arr[distances > cutoff_distance_cartesian] = 0.0

            if not np.any(hop_mat_arr):
                del self.hop[R]
            else:
                self.hop[R] = self._matrix_type(hop_mat_arr)

    def _empty_matrix(self):
        """Returns an empty matrix, either sparse or dense according to the current setting. The size is determined by the system's size"""
        return self._matrix_type(np.zeros((self.size, self.size), dtype=complex))

    def set_sparse(self, sparse: bool = True):
        """
        Defines whether sparse or dense matrices should be used to
        represent the system, and changes the system accordingly if
        needed.

        Parameters
        ----------
        sparse :
            Flag to determine whether the system is set to be sparse
            (``True``) or dense (``False``).
        """
        # check if the right sparsity is alredy set
        # when using from __init__, self._sparse is not set
        with contextlib.suppress(AttributeError):
            if sparse == self._sparse:
                return

        self._sparse = sparse
        if sparse:
            self._matrix_type = sp.csr
        else:
            self._matrix_type = np.array

        # change existing matrices
        with contextlib.suppress(AttributeError):
            for k, v in self.hop.items():
                self.hop[k] = self._matrix_type(v)  # type: ignore

    # If Python 3.4 support is dropped this could be made more straightforwardly
    # However, for now the default pickle protocol (and thus multiprocessing)
    # does not support that.
    def _array_cast(self, x):
        """Casts a matrix type to a numpy array."""
        if self._sparse:
            return np.array(x)
        else:
            return x

    # -------------------CREATING DERIVED MODELS-------------------------#
    # ---- arithmetic operations ----#
    @property
    def _input_kwargs(self):
        return dict(
            hop=self.hop,
            pos=self.pos,
            occ=self.occ,
            uc=self.uc,
            contains_cc=False,
            sparse=self._sparse,
        )

    def symmetrize(
        self,
        symmetries: ty.Sequence[symmetry_representation.SymmetryOperation],
        full_group: bool = False,
        position_tolerance: float = 1e-5,
    ) -> Model:
        """
        Returns a model which is symmetrized w.r.t. the given
        symmetries. This is done by performing a group average over the
        symmetry group.

        Parameters
        ----------
        symmetries :
            Symmetries which the symmetrized model should respect.
        full_group :
            Specifies whether the given symmetries represent the full
            symmetry group, or only a subset from which the full
            symmetry group is generated.
        position_tolerance :
            Absolute tolerance (in reduced coordinates) when matching
            positions after a symmetry has been applied to existing
            positions.
        """
        if full_group:
            new_model = self._apply_operation(
                symmetries[0], position_tolerance=position_tolerance
            )
            return (
                1
                / len(symmetries)
                * sum(
                    (
                        self._apply_operation(s, position_tolerance=position_tolerance)
                        for s in symmetries[1:]
                    ),
                    new_model,
                )
            )
        else:
            new_model = self
            for sym in symmetries:
                order = sym.get_order()
                sym_pow = sym
                tmp_model = new_model
                for _ in range(1, order):
                    tmp_model += (
                        new_model._apply_operation(  # pylint: disable=protected-access
                            sym_pow, position_tolerance=position_tolerance
                        )
                    )
                    sym_pow @= sym
                new_model = 1 / order * tmp_model
            return new_model

    def _apply_operation(  # pylint: disable=too-many-locals
        self, symmetry_operation, position_tolerance
    ) -> Model:
        """
        Helper function to apply a symmetry operation to the model.
        """
        # apply symmetry operation on sublattice positions
        sublattices = self._get_sublattices()

        new_sublattice_pos = [
            symmetry_operation.real_space_operator.apply(latt.pos)
            for latt in sublattices
        ]

        # match to a known sublattice position to determine the shift vector
        uc_shift = []
        for new_pos in new_sublattice_pos:
            nearest_R = np.array(np.rint(new_pos), dtype=int)
            # the new position must be in a neighbouring UC
            valid_shifts = []
            for T in itertools.product(range(-1, 2), repeat=self.dim):
                shift = nearest_R + T
                if any(
                    np.isclose(new_pos - shift, latt.pos, atol=position_tolerance).all()
                    for latt in sublattices
                ):
                    valid_shifts.append(tuple(shift))
            if not valid_shifts:
                raise TbmodelsException(
                    f"New position {new_pos} does not match any known sublattice",
                    exception_marker=SymmetrizeExceptionMarker.POSITIONS_NOT_SYMMETRIC,
                )
            assert (
                len(valid_shifts) == 1
            ), f"New position {new_pos} matches more than one known sublattice"
            uc_shift.append(valid_shifts[0])

        # setting up the indices to slice the hopping matrices
        hop_shifts_idx: ty.Dict[
            ty.Tuple[int, ...], ty.Tuple[ty.List[int], ty.List[int]]
        ] = co.defaultdict(lambda: ([], []))
        for (i, Ti), (j, Tj) in itertools.product(enumerate(uc_shift), repeat=2):
            shift = tuple(np.array(Tj) - np.array(Ti))
            for idx1, idx2 in itertools.product(
                sublattices[i].indices, sublattices[j].indices
            ):
                hop_shifts_idx[shift][0].append(idx1)
                hop_shifts_idx[shift][1].append(idx2)

        # create hoppings with shifted R (by uc_shift[j] - uc_shift[i])
        new_hop: HoppingType = co.defaultdict(self._empty_matrix)
        for R, mat in self.hop.items():
            R_transformed = np.array(
                np.rint(np.dot(symmetry_operation.rotation_matrix, R)),
                dtype=int,
            )
            for shift, (idx1, idx2) in hop_shifts_idx.items():
                new_R = tuple(np.array(R_transformed) + np.array(shift))
                new_hop[new_R][idx1, idx2] += mat[idx1, idx2]

        # apply D(g) ... D(g)^-1 (since D(g) is unitary: D(g)^-1 == D(g)^H)
        for R in new_hop.keys():
            sym_op = np.array(symmetry_operation.repr.matrix).astype(complex)
            if symmetry_operation.repr.has_cc:
                new_hop[R] = np.conj(new_hop[R])
            new_hop[R] = np.dot(
                sym_op, np.dot(new_hop[R], np.conj(np.transpose(sym_op)))
            )

        return Model(**co.ChainMap(dict(hop=new_hop), self._input_kwargs))

    def slice_orbitals(self, slice_idx: ty.List[int]) -> Model:
        """
        Returns a new model with only the orbitals as given in the
        ``slice_idx``. This can also be used to re-order the orbitals.

        Parameters
        ----------
        slice_idx :
            Orbital indices that will be in the resulting model.
        """
        new_pos = self.pos[tuple(slice_idx), :]
        new_hop = {
            key: np.array(val)[np.ix_(slice_idx, slice_idx)]
            for key, val in self.hop.items()
        }
        return Model(**co.ChainMap(dict(hop=new_hop, pos=new_pos), self._input_kwargs))

    @classmethod
    def join_models(cls, *models: Model) -> Model:
        """
        Creates a tight-binding model which contains all orbitals of the
        given input models. The orbitals are ordered by model, such that
        the resulting Hamiltonian is block-diagonal.

        Parameters
        ----------
        models :
            Models which should be joined together.
        """
        if not models:
            raise ValueError("At least one model must be given.")

        first_model = models[0]
        # check dim
        if not _check_compatibility.check_dim(*models):
            raise ValueError("Model dimensions do not match.")
        new_dim = first_model.dim

        # check uc compatibility
        if not _check_compatibility.check_uc(*models):
            raise ValueError("Model unit cells do not match.")
        new_uc = first_model.uc

        # join positions (must either all be set, or all None)
        pos_list = list(m.pos for m in models)
        # Note: this is affected by issue #76
        if any(pos is None for pos in pos_list):
            if not all(pos is None for pos in pos_list):
                raise ValueError("Either all or no positions must be set.")
            new_pos = None
        else:
            new_pos = np.concatenate(pos_list)

        # add occ (is set to None if any model has occ=None)
        occ_list = list(m.occ for m in models)
        if any(occ is None for occ in occ_list):
            new_occ = None
        else:
            new_occ = sum(occ_list)

        # combine hop
        all_R: ty.Set[ty.Tuple[int, ...]] = set()
        for m in models:
            all_R.update(m.hop.keys())

        new_hop = dict()

        for R in all_R:
            hop_list = [np.array(m.hop[R]) for m in models]
            new_hop[R] = la.block_diag(*hop_list)

        return cls(
            dim=new_dim,
            uc=new_uc,
            pos=new_pos,
            occ=new_occ,
            hop=new_hop,
            contains_cc=False,
        )

    def change_unit_cell(  # pylint: disable=too-many-branches
        self,
        *,
        uc: ty.Optional[ty.Sequence[ty.Sequence[float]]] = None,
        offset: ty.Sequence[float] = (0, 0, 0),
        cartesian: bool = False,
    ) -> Model:
        """Return a model with a different unit cell of the same volume.

        Creates a model with a changed unit cell - with a different
        shape and / or origin. The new unit cell must be compatible
        with the current lattice, and have the same volume.

        Parameters
        ----------
        uc :
            The new unit cell shape. Lattice vectors are given as rows
            in a (dim x dim) matrix. If no unit cell is given, the
            current unit cell shape is kept.
        offset :
            The position of the new unit cell origin, relative to the old
            one.
        cartesian :
            Specifies if the offset and unit cell are in cartesian or
            reduced coordinates. Reduced coordinates are with respect to
            the *old* unit cell.
        """
        # Validate inputs w.r.t. model properties
        # Note: this is affected by issue #76
        if self.pos is None:
            raise ValueError(
                "Cannot change the unit cell: model positions are not defined."
            )

        if cartesian:
            if self.uc is None:
                raise ValueError(
                    "Cannot change unit cell in cartesian coordinates: model does not have a unit cell."
                )
            # convert to reduced coordinates
            if uc is None:
                new_uc: ty.Optional[np.ndarray] = self.uc
                uc_reduced = np.eye(self.dim)
            else:
                new_uc = np.array(uc)
                uc_reduced = la.solve(self.uc.T, new_uc.T).T
            offset_reduced = la.solve(self.uc.T, np.array(offset).T).T
        else:
            if uc is None:
                uc_reduced = np.eye(self.dim)
            else:
                uc_reduced = np.array(uc)
            if self.uc is None:
                new_uc = None
            else:
                new_uc = (self.uc.T @ uc_reduced.T).T
            offset_reduced = np.array(offset)

        # check that the reduced unit cell is compatible with the
        # current lattice
        if not np.allclose(np.round(uc_reduced), uc_reduced):
            raise ValueError(
                "The new unit cell must be compatible with the current lattice. "
                "It must be an integer combination of previous lattice vectors, "
                f"but in reduced coordinates it is:\n{uc_reduced}"
            )
        uc_reduced = np.round(uc_reduced).astype(int)
        if la.det(uc_reduced) != 1:
            raise ValueError(
                "The determinant of the unit cell in reduced coordinates must "
                f"be 1, but it is {la.det(uc_reduced)} instead."
            )

        # apply offset to positions
        new_pos = self.pos - offset_reduced

        # rotate positions
        new_pos = la.solve(uc_reduced.T, new_pos.T).T

        # rotate hopping matrices
        new_hop = {}
        for R, hop_mat in self.hop.items():
            new_R = la.solve(uc_reduced.T, R)
            assert np.allclose(np.round(new_R), new_R)
            new_R = tuple(np.round(new_R).astype(int))
            new_hop[new_R] = hop_mat

        return Model(
            **co.ChainMap(dict(uc=new_uc, pos=new_pos, hop=new_hop), self._input_kwargs)
        )

    def supercell(  # pylint: disable=too-many-locals
        self, size: ty.Sequence[int]
    ) -> Model:
        """Generate a model for a supercell of the current unit cell.

        Parameters
        ----------
        size :
            The size of the supercell, given as integer multiples of the
            current lattice vectors
        """
        size_array = np.array(size).astype(dtype=int, casting="safe")
        if size_array.shape != (self.dim,):
            raise ValueError(
                "The given 'size' has incorrect shape {}, should be {}.".format(
                    size_array.shape, (self.dim,)
                )
            )
        volume_multiplier = np.prod(size_array)
        new_occ = None if self.occ is None else volume_multiplier * self.occ
        if self.uc is None:
            new_uc = None
        else:
            new_uc = (self.uc.T * size_array).T

        # the new positions, normalized to the supercell
        new_pos: ty.List[np.ndarray] = []
        reduced_pos = np.array([p / size_array for p in self.pos])
        uc_offsets = list(
            np.array(offset)
            for offset in itertools.product(*[range(n) for n in size_array])
        )
        for current_uc_offset in uc_offsets:
            new_pos.extend(reduced_pos + (current_uc_offset / size_array))

        new_size = self.size * volume_multiplier
        new_hop: HoppingType = co.defaultdict(
            lambda: np.zeros((new_size, new_size), dtype=complex)
        )

        # Can be used to get the orbital offset of a given unit cell
        # by taking the inner product with the unit cell position.
        uc_idx_multiplier = (
            np.array([np.prod(size[i:], dtype=int) for i in range(1, len(size) + 1)])
            * self.size
        )

        for uc1_idx, uc1_pos in enumerate(uc_offsets):
            uc1_idx_offset = uc1_idx * self.size

            for R, hop_mat in self.hop.items():
                hop_mat = self._array_cast(hop_mat)

                # position of the uc of orbital 2, not mapped inside supercell
                full_uc2_pos = uc1_pos + R
                # mapped into the supercell
                uc2_pos = full_uc2_pos % size_array
                uc2_idx_offset = np.inner(uc_idx_multiplier, uc2_pos)

                # R in terms of supercells
                new_R = np.array(np.floor(full_uc2_pos / size_array), dtype=int)

                new_hop[tuple(new_R)][
                    uc1_idx_offset : uc1_idx_offset + self.size,
                    uc2_idx_offset : uc2_idx_offset + self.size,
                ] += hop_mat

        return Model(
            **co.ChainMap(
                dict(
                    hop=new_hop,
                    occ=new_occ,
                    uc=new_uc,
                    size=new_size,
                    pos=new_pos,
                    contains_cc=False,
                ),
                self._input_kwargs,
            )
        )

    def fold_model(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self,
        *,
        new_unit_cell: ty.Sequence[ty.Sequence[float]],
        unit_cell_offset: ty.Sequence[float] = (0, 0, 0),
        position_tolerance: float = 1e-3,
        unmatched_position_threshold: float = float("inf"),
        orbital_labels: ty.Sequence[ty.Hashable],
        target_indices: ty.Optional[ty.Sequence[int]] = None,
        check_cc: bool = True,
        check_uc_volume: bool = True,
        uc_volume_tolerance: float = 1e-6,
        check_orbital_ratio: bool = True,
        order_by: Literal["label", "index"] = "label",
    ) -> Model:
        """
        Returns a model with a smaller unit cell. Orbitals which are related
        by a lattice vector of the new unit cell are "folded" into a single
        orbital.
        This is the inverse operation of the supercell construction.

        .. note :: This function is currently experimental, and its interface
            may still change without warning.

        Parameters
        ----------
        new_unit_cell :
            The unit cell of the new model. Note that this unit cell must
            lie completely within the old unit cell.
        unit_cell_offset :
            Position (in cartesian coordinates) at which the new unit cell
            is based.
        position_tolerance :
            Tolerance used when determining if a position mapped into the
            new unit cell is the same as an existing orbital position.
        unmatched_position_threshold :
            Threshold above which orbital positions that do not have a
            matching position in the new model are ignored. The
            threshold is defined as cartesian distance from the new unit
            cell origin.
        orbital_labels :
            A list of labels for the orbitals of the current model. This
            is needed to distinguish between orbitals of the same position
            when mapping them to the new orbitals.
        target_indices :
            Optional list of indices (in the current model) of the orbitals
            which lie within the new unit cell. This can be used as a
            check that the new orbitals are the expected ones.
        check_cc :
            Flag to determine whether the hoppings should be checked for
            being perfect complex conjugates. This check can fail if the
            model does not have exact translation symmetry w.r.t. the new
            unit cell. If set to false, hoppings are averaged to be
            complex conjugates.
        check_uc_volume :
            Flag to determine if the unit cell volume should decrease
            by the same factor as the number of orbitals.
        uc_volume_tolerance :
            Absolute tolerance when checking if the unit cell volume change
            is consistent with the change in number of orbitals.
        check_orbital_ratio :
            Flag to determine if the ratio of individual orbital labels
            should be checked to be the same as the initial ratio. If
            this is set to False, the resulting model will always
            have ``occ=None``.
        order_by :
            Determines how the orbitals in the new model are ordered.
            For ``order_by="index"``, the orbitals are ordered exactly the
            same as in the original model. Note that this order will
            depend on which orbitals end up inside the unit cell.
            For ``order_by="label"``, the orbitals are ordered
            according to the **occurrence** (not the value) of the
            labels in the ``orbital_labels`` input. Orbitals with the
            same label are again ordered by index.
        """
        if len(orbital_labels) != self.size:
            raise ValueError(
                f"The lenght of the 'orbital_labels' input ({len(orbital_labels)}) "
                f"does not match the size of the model ({self.size})."
            )
        # Note: this is affected by issue #76
        if self.uc is None or self.pos is None:
            raise ValueError(
                "Unit cell and positions must be specified for model folding."
            )

        new_uc = np.array(new_unit_cell)
        # Check that the new unit cell lies within the current one
        new_uc_vertices = unit_cell_offset + np.array(
            [
                sum(mult * a_i for mult, a_i in zip(multipliers, new_uc))
                for multipliers in itertools.product([0, 1], repeat=self.dim)
            ]
        )
        new_uc_vertices_reduced = la.solve(self.uc.T, new_uc_vertices.T).T
        eps = 1e-6
        if not np.all(
            np.logical_and(
                new_uc_vertices_reduced >= -eps, new_uc_vertices_reduced <= 1 + eps
            )
        ):
            raise ValueError(
                "The new unit cell is not contained within the current one. The new "
                "unit cell vertices in reduced coordinates are:\n"
                f"{new_uc_vertices_reduced}."
            )

        positions_cartesian = (self.uc.T @ self.pos.T).T
        pos_cartesian_relative = positions_cartesian - unit_cell_offset
        pos_reduced_new = la.solve(new_uc.T, pos_cartesian_relative.T).T

        # Check and warn if positions are at the edge of the new unit cell.
        at_uc_edge_indices = list(
            np.argwhere(
                np.logical_and(
                    np.any(
                        np.logical_or(
                            np.isclose(pos_reduced_new, 0, rtol=0, atol=2 * eps),
                            np.isclose(pos_reduced_new, 1, rtol=0, atol=2 * eps),
                        ),
                        axis=-1,
                    ),
                    np.all(
                        np.logical_and(
                            pos_reduced_new >= -2 * eps,
                            pos_reduced_new <= 1 + 2 * eps,
                        ),
                        axis=-1,
                    ),
                )
            ).flatten()
        )
        if at_uc_edge_indices:
            warnings.warn(
                f"The positions of the orbitals with indices {at_uc_edge_indices}, are "
                "close to the border of the new unit cell (new reduced coordinates "
                f" {pos_reduced_new[at_uc_edge_indices]}). This can lead to incorrect "
                "classification of positions inside / outside the new unit cell, "
                "which leads to an incorrect model.",
                UserWarning,
            )

        in_uc_indices = np.argwhere(
            np.all(
                np.logical_and(pos_reduced_new >= 0, pos_reduced_new < 1),
                axis=-1,
            )
        ).flatten()
        if order_by == "label":
            idx = 0
            orbital_sort_idx = {}
            for label in orbital_labels:
                if label not in orbital_sort_idx:
                    orbital_sort_idx[label] = idx
                    idx += 1
            in_uc_sort_idx = [
                orbital_sort_idx[orbital_labels[i]] for i in in_uc_indices
            ]
            in_uc_indices = in_uc_indices[
                np.argsort(in_uc_sort_idx, kind="mergesort")  # need stable sorting
            ]
        else:
            if order_by != "index":
                raise ValueError(
                    f"Invalid input '{order_by}' for 'order_by', must be either 'label' or 'index'."
                )
        if target_indices is not None:
            if not np.all(target_indices == in_uc_indices):
                raise ValueError(
                    f"The indices for atoms in the given unit cell ({in_uc_indices}) do not match the target indices ({target_indices})."
                )

        # Check that all orbitals are present in the correct number
        total_orbital_ratio = len(self.pos) / len(in_uc_indices)
        if check_orbital_ratio:
            orbital_counts_initial = co.Counter(orbital_labels)
            orbital_counts_new: ty.Counter[ty.Hashable] = co.Counter(
                np.array(orbital_labels)[np.ix_(in_uc_indices)]
            )

            for label, count_initial in orbital_counts_initial.items():
                count_new = orbital_counts_new.get(label, 0)
                if not np.isclose(1 / total_orbital_ratio, count_new / count_initial):
                    raise ValueError(
                        "The individual orbital numbers inside the new unit cell "
                        "are not consistent with the change in total orbital number:\n"
                        f"Total: {len(in_uc_indices)}/{len(self.pos)}\n"
                        + "\n".join(
                            f"Orbital '{key}': { orbital_counts_new.get(key, 0)}/{orbital_counts_initial[key]}"
                            for key in orbital_counts_initial.keys()
                        )
                    )
        if check_uc_volume:
            uc_volume_change_factor = la.det(self.uc) / la.det(new_uc)
            if not np.isclose(
                uc_volume_change_factor,
                total_orbital_ratio,
                atol=uc_volume_tolerance,
            ):
                raise ValueError(
                    f"The unit cell volume decreased by a factor {uc_volume_change_factor}, "
                    "which is inconsistent with the decrease in the number of "
                    f"orbitals (factor {total_orbital_ratio})."
                )

        new_pos = pos_reduced_new[np.ix_(in_uc_indices)]
        in_uc_labels = np.array(orbital_labels)[np.ix_(in_uc_indices)]

        new_occ: ty.Optional[int]
        if check_orbital_ratio and self.occ is not None:
            new_occ_float = self.occ / total_orbital_ratio
            new_occ = int(np.round(new_occ_float))
            if not np.isclose(new_occ, new_occ_float):
                raise ValueError(
                    f"The occupation number {new_occ_float} of the resulting model is fractional."
                )
        else:
            new_occ = None

        offset_stencil = np.array(
            list(itertools.product(*[range(-1, 2) for _ in range(self.dim)]))
        )

        def get_min_distance_and_offset(pos1, pos2):
            """
            pos1 and pos2 both reduced, within [0, 1)
            offset in reduced, added to pos2
            """

            diff = pos2 - pos1
            total_diffs = offset_stencil + diff
            distances = la.norm(new_uc.T @ total_diffs.T, axis=0)
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]
            min_offset = offset_stencil[np.argmin(distances)]
            return min_dist, min_offset

        def get_matching_idx_and_offset(pos_reduced, orbital_label):
            res_idx = []
            main_offset = np.array(np.floor(pos_reduced), dtype=int)
            pos_relative = pos_reduced % 1
            assert np.allclose(main_offset + pos_relative, pos_reduced)
            for i, (pos, target_label) in enumerate(zip(new_pos, in_uc_labels)):
                if orbital_label == target_label:
                    dist, curr_offset = get_min_distance_and_offset(
                        pos1=pos, pos2=pos_relative
                    )

                    if dist < position_tolerance:
                        res_idx.append((i, main_offset - curr_offset))

            if len(res_idx) == 1:
                return res_idx[0]
            elif len(res_idx) > 1:
                raise ValueError(f"More than one matching index found: {res_idx}.")
            else:
                pos_dist = la.norm(new_uc.T @ pos_reduced)
                if pos_dist < unmatched_position_threshold:
                    raise ValueError(
                        "The orbital at position (in new reduced coordinates) "
                        f"{pos_reduced} does not match any orbital in the new model."
                    )
                return None, None

        new_size = len(in_uc_labels)
        new_hop: HoppingType = co.defaultdict(
            lambda: np.zeros((new_size, new_size), dtype=complex)
        )
        for input_R, input_mat in self.hop.items():
            R_reduced_to_new_uc = la.solve(new_uc.T, (self.uc.T @ input_R))
            input_mat = self._array_cast(input_mat)
            # Construct the new hoppings from the full set of current
            # hoppings, because +R and -R do not necessarily map to the
            # same new current_offset.
            for effective_R, effective_input_mat in [
                (R_reduced_to_new_uc, input_mat),
                (-R_reduced_to_new_uc, input_mat.conjugate().transpose()),
            ]:
                if not check_cc:
                    effective_input_mat = effective_input_mat / 2
                for orbital_2_idx in range(self.size):
                    full_reduced_pos_2 = effective_R + pos_reduced_new[orbital_2_idx]
                    new_2_idx, current_offset = get_matching_idx_and_offset(
                        full_reduced_pos_2, orbital_labels[orbital_2_idx]
                    )
                    if new_2_idx is not None:
                        new_hop[tuple(current_offset)][
                            :, new_2_idx
                        ] += effective_input_mat[in_uc_indices, orbital_2_idx]

        return Model(
            **co.ChainMap(
                dict(
                    hop=new_hop,
                    size=new_size,
                    pos=new_pos,
                    uc=new_uc,
                    contains_cc=check_cc,
                    occ=new_occ,
                ),
                self._input_kwargs,
            )
        )

    def __add__(self, model: Model) -> Model:
        """
        Adds two models together by adding their hopping terms.
        """
        if not isinstance(model, Model):
            raise ValueError(
                "Invalid argument type for Model.__add__: {}".format(type(model))
            )

        # ---- CONSISTENCY CHECKS ----
        # check if the occupation number matches
        if self.occ != model.occ:
            raise ValueError(
                "Error when adding Models: occupation numbers ({}, {}) don't match".format(
                    self.occ, model.occ
                )
            )

        # check if the size of the hopping matrices match
        if self.size != model.size:
            raise ValueError(
                "Error when adding Models: the number of states ({}, {}) doesn't match".format(
                    self.size, model.size
                )
            )

        # check if the unit cells match
        if not _check_compatibility.check_uc(self, model):
            raise ValueError(
                "Error when adding Models: unit cells don't match.\nModel 1:\n{0.uc}\n\nModel 2:\n{1.uc}".format(
                    self, model
                )
            )

        # check if the positions match
        pos_match = True
        tolerance = 1e-6
        for v1, v2 in zip(self.pos, model.pos):
            if not pos_match:
                break
            for x1, x2 in zip(v1, v2):
                if abs(x1 - x2) > tolerance:
                    pos_match = False
                    break
        if not pos_match:
            raise ValueError(
                "Error when adding Models: positions don't match.\nModel 1:\n{0.pos}\n\nModel 2:\n{1.pos}".format(
                    self, model
                )
            )

        # ---- MAIN PART ----
        new_hop = copy.deepcopy(self.hop)
        for R, hop_mat in model.hop.items():
            new_hop[R] += hop_mat
        # -------------------
        return Model(**co.ChainMap(dict(hop=new_hop), self._input_kwargs))

    def __sub__(self, model: Model) -> Model:
        """
        Substracts one model from another by substracting all hopping terms.
        """
        return self + -model

    def __neg__(self) -> Model:
        """
        Changes the sign of all hopping terms.
        """
        return -1 * self

    def __mul__(self, x: float) -> Model:
        """
        Multiplies hopping terms by x.
        """
        new_hop = dict()
        for R, hop_mat in self.hop.items():
            new_hop[R] = x * hop_mat

        return Model(**co.ChainMap(dict(hop=new_hop), self._input_kwargs))

    def __rmul__(self, x: float) -> Model:
        """
        Multiplies hopping terms by x.
        """
        return self.__mul__(x)

    def __truediv__(self, x: float) -> Model:
        """
        Divides hopping terms by x.
        """
        return self * (1.0 / x)
