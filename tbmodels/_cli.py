#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>

import os
from collections.abc import Iterable
from functools import singledispatch

import click
import symmetry_representation as sr

from ._tb_model import Model

@click.group()
def cli():
    pass

@cli.command(short_help='Parse Wannier90 output files to an HDF5 file.')
@click.option(
    '--folder', '-f',
    type=click.Path(exists=True, file_okay=False),
    default='.',
    help='Directory containing the Wannier90 output files.'
)
@click.option(
    '--prefix', '-p',
    type=str,
    default='wannier',
    help='Common prefix of the Wannier90 output files.'
)
@click.option(
    '--output', '-o',
    type=click.Path(),
    default='model.hdf5',
    help='Path of the output file.'
)
def parse(folder, prefix, output):
    """
    Parse Wannier90 output files and create an HDF5 file containing the tight-binding model.
    """
    click.echo("Parsing output files '{}*' ...".format(os.path.join(folder, prefix)))
    model = Model.from_wannier_folder(folder=folder, prefix=prefix, ignore_orbital_order=True)
    click.echo("Writing model to file '{}' ...".format(output))
    model.to_hdf5_file(output)
    click.echo("Done!")


@cli.command()
@click.option(
    '--input', '-i',
    type=click.Path(exists=True, dir_okay=False),
    default='model.hdf5',
    help='File containing the model that will be symmetrized.'
)
@click.option(
    '--output', '-o',
    type=click.Path(dir_okay=False),
    default='model_symmetrized.hdf5',
    help='Output file for the symmetrized model.'
)
@click.option(
    '--symmetries', '-s',
    type=click.Path(),
    help='File containing symmetry_representation.SymmetryGroup objects (in HDF5 form).'
)
@click.option(
    '--full-group/--no-full-group', '-f/-nf',
    default=None,
    help="""
    Full group: The full symmetry group is given in the symmetries.
    No full group: The symmetries only contain a generating subset of the full group. Overrides the option given in the symmetries file (if any).
    """
)
def symmetrize(input, output, symmetries, full_group):
    click.echo("Reading initial model from file '{}' ...".format(input))
    model = Model.from_hdf5_file(input)
    click.echo("Reading symmetries from file '{}' ...".format(symmetries))
    sym = sr.io.load(symmetries)
    model_sym = _symmetrize(sym, model, full_group)
    click.echo("Writing symmetrized model to file '{}' ...".format(output))
    model_sym.to_hdf5_file(output)
    click.echo('Done!')

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
    click.echo("Symmetrizing model with {} symmetr{}, full_group={} ...".format(
        len(symmetries), 'y' if len(symmetries) == 1 else 'ies', full_group
    ))
    return model.symmetrize(symmetries=symmetries, full_group=full_group)

@_symmetrize.register(sr.SymmetryOperation)
def _(sym, model, full_group):
    sym_group = sr.SymmetryGroup(
        symmetries=[sym],
        full_group=full_group or False # catches 'None', does nothing for 'True' or 'False'
    )
    return _symmetrize(sym_group, model, full_group)
