#
# (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""Defines the tbmodels command-line interface."""

import os
import enum
from collections.abc import Iterable
from functools import singledispatch

import click

from . import __version__ as tbmodels_version
from . import Model
from .exceptions import TbmodelsException, SymmetrizeExceptionMarker


@click.group()
@click.version_option(version=tbmodels_version)
def cli():
    pass


def _get_output_option(**kwargs):
    return click.option("--output", "-o", type=click.Path(dir_okay=False), **kwargs)


class _SparsityChoices(enum.Enum):
    SPARSE = "sparse"
    DENSE = "dense"
    AS_INPUT = "as_input"


_SPARSITY_OPTION = click.option(
    "--sparsity",
    default=_SparsityChoices.AS_INPUT.value,
    type=click.Choice([choice.value for choice in _SparsityChoices]),
    help="Write the model in sparse format. By default, the format of the input model is used.",
)

_INPUT_OPTION = click.option(
    "--input",
    "-i",
    type=click.Path(exists=True, dir_okay=False),
    default="model.hdf5",
    help="File containing the input model (in HDF5 format).",
)

_VERBOSE_OPTION = click.option("-v", "--verbose", is_flag=True, default=False)


def _read_input(input, verbose):  # pylint: disable=redefined-builtin
    if verbose:
        click.echo(f"Reading initial model from file '{input}' ...")
    return Model.from_hdf5_file(input)


def _write_output(model, output, sparsity, verbose):
    """
    Helper function to write a tight-binding model to an output file.
    The sparsity of the model is changed if specified through the
    'sparsity' option.
    """
    assert sparsity in [choice.value for choice in _SparsityChoices]
    if sparsity == _SparsityChoices.SPARSE.value:
        model.set_sparse(True)
    elif sparsity == _SparsityChoices.DENSE.value:
        model.set_sparse(False)
    if verbose:
        click.echo(f"Writing output model to file '{output}' ...")
    model.to_hdf5_file(output)
    if verbose:
        click.echo("Done!")


@cli.command(short_help="Parse Wannier90 output files to an HDF5 file.")
@click.option(
    "--folder",
    "-f",
    type=click.Path(exists=True, file_okay=False),
    default=".",
    help="Directory containing the Wannier90 output files.",
)
@click.option(
    "--prefix",
    "-p",
    type=str,
    default="wannier",
    help="Common prefix of the Wannier90 output files.",
)
@click.option(
    "--pos-kind",
    type=click.Choice(["wannier", "nearest_atom"]),
    default="wannier",
    help="Which position to use for the orbitals.",
)
@click.option("--distance-ratio-threshold", type=click.FloatRange(min=1.0), default=3.0)
@_SPARSITY_OPTION  # pylint: disable=too-many-arguments
@_VERBOSE_OPTION
@_get_output_option(default="model.hdf5", help="Path of the output file.")
def parse(
    folder,
    prefix,
    output,
    pos_kind,
    distance_ratio_threshold,
    sparsity,
    verbose,
):
    """
    Parse Wannier90 output files and create an HDF5 file containing the tight-binding model.
    """
    if verbose:
        click.echo(
            "Parsing output files '{}*' ...".format(os.path.join(folder, prefix))
        )
    model = Model.from_wannier_folder(
        folder=folder,
        prefix=prefix,
        ignore_orbital_order=True,
        pos_kind=pos_kind,
        distance_ratio_threshold=distance_ratio_threshold,
    )
    _write_output(model, output, sparsity=sparsity, verbose=verbose)


@cli.command(short_help="Create symmetrized tight-binding model.")
@_INPUT_OPTION
@_get_output_option(
    default="model_symmetrized.hdf5",
    help="Output file for the symmetrized model.",
)
@click.option(
    "--symmetries",
    "-s",
    type=click.Path(),
    default="symmetries.hdf5",
    help="File containing symmetry_representation.SymmetryGroup objects (in HDF5 form).",
)
@click.option(
    "--full-group/--no-full-group",
    "-f",
    default=None,
    help="""
    Full group: The full symmetry group is given in the symmetries.
    No full group: The symmetries only contain a generating subset of the full group. Overrides the option given in the symmetries file (if any).
    """,
)
@_SPARSITY_OPTION
@_VERBOSE_OPTION
def symmetrize(
    input, output, symmetries, full_group, sparsity, verbose
):  # pylint: disable=redefined-builtin
    """Symmetrize tight-binding model with given symmetry group(s)."""
    import symmetry_representation as sr  # pylint: disable=import-outside-toplevel

    model = _read_input(input, verbose=verbose)
    if verbose:
        click.echo(f"Reading symmetries from file '{symmetries}' ...")
    sym = sr.io.load(symmetries)

    @singledispatch
    def _symmetrize(sym, model, full_group):
        """
        Implementation for the symmetrization procedure. The singledispatch is used
        to treat (nested) lists of symmetries or symmetry groups.
        """
        raise TbmodelsException(
            f"The given symmetries object has invalid type '{type(sym)}'",
            SymmetrizeExceptionMarker.INVALID_SYMMETRY_TYPE,
        )

    @_symmetrize.register(Iterable)
    def _(sym, model, full_group):
        for s in sym:
            model = _symmetrize(s, model, full_group)
        return model

    @_symmetrize.register(sr.SymmetryGroup)
    def _(sym, model, full_group):
        symmetries = sym.symmetries
        if full_group is None:
            full_group = sym.full_group
        click.echo(
            "Symmetrizing model with {} symmetr{}, full_group={} ...".format(
                len(symmetries),
                "y" if len(symmetries) == 1 else "ies",
                full_group,
            )
        )
        return model.symmetrize(symmetries=symmetries, full_group=full_group)

    @_symmetrize.register(sr.SymmetryOperation)
    def _(sym, model, full_group):
        sym_group = sr.SymmetryGroup(
            symmetries=[sym],
            full_group=full_group
            or False,  # catches 'None', does nothing for 'True' or 'False'
        )
        return _symmetrize(sym_group, model, full_group)

    model_sym = _symmetrize(sym, model, full_group)
    _write_output(model_sym, output, sparsity=sparsity, verbose=verbose)


@cli.command(short_help="Slice specific orbitals from model.")
@_INPUT_OPTION
@_get_output_option(
    default="model_sliced.hdf5", help="Output file for the sliced model."
)
@click.argument(
    "slice-idx",
    type=int,
    nargs=-1,
)  # pylint: disable=redefined-builtin
@_SPARSITY_OPTION
@_VERBOSE_OPTION
def slice(
    input, output, slice_idx, sparsity, verbose
):  # pylint: disable=redefined-builtin
    """
    Create a model containing only the orbitals given in the SLICE_IDX.
    """
    model = _read_input(input, verbose=verbose)
    if verbose:
        click.echo(f"Slicing model with indices {slice_idx} ...")
    model_slice = model.slice_orbitals(slice_idx=slice_idx)
    _write_output(model_slice, output, sparsity=sparsity, verbose=verbose)


@cli.command(short_help="Calculate energy eigenvalues.")
@_INPUT_OPTION
@click.option(
    "-k",
    "--kpoints",
    type=click.Path(exists=True, dir_okay=False),
    default="kpoints.hdf5",
    help="File containing the k-points for which the eigenvalues are evaluated.",
)
@_get_output_option(
    default="eigenvals.hdf5", help="Output file for the energy eigenvalues."
)
@_VERBOSE_OPTION
def eigenvals(input, kpoints, output, verbose):  # pylint: disable=redefined-builtin
    """
    Calculate the energy eigenvalues for a given set of k-points (in reduced coordinates). The input and output is given in an HDF5 file.
    """
    import bands_inspect as bi  # pylint: disable=import-outside-toplevel

    model = _read_input(input, verbose=verbose)
    if verbose:
        click.echo(f"Reading kpoints from file '{kpoints}' ...")
    kpts = bi.io.load(kpoints)
    if isinstance(kpts, bi.eigenvals.EigenvalsData):
        kpts = kpts.kpoints

    if verbose:
        click.echo("Calculating energy eigenvalues ...")
    eigenvalues = bi.eigenvals.EigenvalsData.from_eigenval_function(
        kpoints=kpts, eigenval_function=model.eigenval, listable=True
    )
    if verbose:
        click.echo(f"Writing kpoints and energy eigenvalues to file '{output}' ...")
    bi.io.save(eigenvalues, output)
    if verbose:
        click.echo("Done!")
