#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>

import os

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
    help='File containing a single symmetry_representation.SymmetryGroup (in HDF5 form).'
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
    click.echo("Reading symmetry group from file '{}' ...".format(symmetries))
    sym_group = sr.io.load(symmetries)
    symmetries = sym_group.symmetries
    full_group = full_group if full_group is not None else sym_group.full_group
    click.echo("Symmetrizing model with {} symmetries, full_group={} ...".format(
        len(symmetries), full_group)
    )
    model_sym = model.symmetrize(symmetries=symmetries, full_group=full_group)
    click.echo("Writing symmetrized model to file '{}' ...".format(output))
    model_sym.to_hdf5_file(output)
    click.echo('Done!')
