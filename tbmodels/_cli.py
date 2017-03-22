#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>

import os

import click

from ._tb_model import Model

@click.group()
def cli():
    pass

@cli.command()
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
    type=click.Path(),
    default='model_symmetrized.hdf5',
    help='Output file for the symmetrized model.'
)
@click.option(
    '--symmetries', '-s',
    type=click.Path(),
    help='File containing the symmetry operations in JSON form.'
)
def symmetrize(input, output, symmetries):
    click.echo('Symmetrize function')
    raise NotImplementedError
