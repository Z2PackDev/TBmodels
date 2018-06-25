import os
from collections.abc import Iterable
from functools import singledispatch

import click
import numpy as np
import bands_inspect as bi
import symmetry_representation as sr

from . import Model


@click.group()
def cli():
    pass


def _output_option(**kwargs):
    return click.option('--output', '-o', type=click.Path(dir_okay=False), **kwargs)


_input_option = click.option(
    '--input',
    '-i',
    type=click.Path(exists=True, dir_okay=False),
    default='model.hdf5',
    help='File containing the input model (in HDF5 format).'
)


def _read_input(input):
    click.echo("Reading initial model from file '{}' ...".format(input))
    return Model.from_hdf5_file(input)


def _write_output(model, output):
    click.echo("Writing output model to file '{}' ...".format(output))
    model.to_hdf5_file(output)
    click.echo("Done!")


@cli.command(short_help='Parse Wannier90 output files to an HDF5 file.')
@click.option(
    '--folder',
    '-f',
    type=click.Path(exists=True, file_okay=False),
    default='.',
    help='Directory containing the Wannier90 output files.'
)
@click.option('--prefix', '-p', type=str, default='wannier', help='Common prefix of the Wannier90 output files.')
@click.option(
    '--pos-kind',
    type=click.Choice(['wannier', 'nearest_atom']),
    default='wannier',
    help="Which position to use for the orbitals."
)
@_output_option(default='model.hdf5', help='Path of the output file.')
def parse(folder, prefix, output, pos_kind):
    """
    Parse Wannier90 output files and create an HDF5 file containing the tight-binding model.
    """
    click.echo("Parsing output files '{}*' ...".format(os.path.join(folder, prefix)))
    model = Model.from_wannier_folder(folder=folder, prefix=prefix, ignore_orbital_order=True, pos_kind=pos_kind)
    _write_output(model, output)


@cli.command(short_help='Create symmetrized tight-binding model.')
@_input_option
@_output_option(default='model_symmetrized.hdf5', help='Output file for the symmetrized model.')
@click.option(
    '--symmetries',
    '-s',
    type=click.Path(),
    default='symmetries.hdf5',
    help='File containing symmetry_representation.SymmetryGroup objects (in HDF5 form).'
)
@click.option(
    '--full-group/--no-full-group',
    '-f',
    default=None,
    help="""
    Full group: The full symmetry group is given in the symmetries.
    No full group: The symmetries only contain a generating subset of the full group. Overrides the option given in the symmetries file (if any).
    """
)
def symmetrize(input, output, symmetries, full_group):
    """
    Symmetrize tight-binding model with given symmetry group(s).
    """
    model = _read_input(input)
    click.echo("Reading symmetries from file '{}' ...".format(symmetries))
    sym = sr.io.load(symmetries)
    model_sym = _symmetrize(sym, model, full_group)
    _write_output(model_sym, output)


@singledispatch
def _symmetrize(sym, model, full_group):
    raise ValueError("Invalid type '{}' for _symmetrize".format(type(sym)))


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
            len(symmetries), 'y' if len(symmetries) == 1 else 'ies', full_group
        )
    )
    return model.symmetrize(symmetries=symmetries, full_group=full_group)


@_symmetrize.register(sr.SymmetryOperation)
def _(sym, model, full_group):
    sym_group = sr.SymmetryGroup(
        symmetries=[sym],
        full_group=full_group or False  # catches 'None', does nothing for 'True' or 'False'
    )
    return _symmetrize(sym_group, model, full_group)


@cli.command(short_help="Slice specific orbitals from model.")
@_input_option
@_output_option(default='model_sliced.hdf5', help='Output file for the sliced model.')
@click.argument(
    'slice-idx',
    type=int,
    nargs=-1,
)
def slice(input, output, slice_idx):
    """
    Create a model containing only the orbitals given in the SLICE_IDX.
    """
    model = _read_input(input)
    click.echo("Slicing model with indices {} ...".format(slice_idx))
    model_slice = model.slice_orbitals(slice_idx=slice_idx)
    _write_output(model_slice, output)


@cli.command(short_help="Calculate energy eigenvalues.")
@_input_option
@click.option(
    '-k',
    '--kpoints',
    type=click.Path(exists=True, dir_okay=False),
    default='kpoints.hdf5',
    help='File containing the k-points for which the eigenvalues are evaluated.'
)
@_output_option(default='eigenvals.hdf5', help='Output file for the energy eigenvalues.')
def eigenvals(input, kpoints, output):
    """
    Calculate the energy eigenvalues for a given set of k-points (in reduced coordinates). The input and output is given in an HDF5 file.
    """
    model = _read_input(input)
    click.echo("Reading kpoints from file '{}' ...".format(kpoints))
    kpts = bi.io.load(kpoints)

    click.echo("Calculating energy eigenvalues ...")
    eigenvalues = bi.eigenvals.EigenvalsData.from_eigenval_function(kpoints=kpts, eigenval_function=model.eigenval)

    click.echo("Writing kpoints and energy eigenvalues to file '{}' ...".format(output))
    bi.io.save(eigenvalues, output)
    click.echo("Done!")
