#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>

import click

from ._tb_model import Model

@click.group()
def cli():
    pass
    
@cli.command()
def parse():
    click.echo('Parse function')
