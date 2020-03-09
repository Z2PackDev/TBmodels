# -*- coding: utf-8 -*-
"""
Defines custom exceptions to be used in TBmodels. The main purpose of
these custom exceptions is their use in the CLI. Custom exceptions can
be caught in the CLI functions.
"""

from enum import Enum

import click

__all__ = ('TbmodelsException', 'ExceptionMarker', 'ParseExceptionMarker')


class ExceptionMarker(Enum):
    """
    A list of markers to be used in custom TBmodels exceptions.
    """


class TbmodelsException(click.ClickException):
    """Base class for the custom TBmodels exceptions."""
    def __init__(self, msg, exception_marker: ExceptionMarker):
        formatted_msg = f'[{exception_marker.name}] {msg}'
        super().__init__(formatted_msg)


class ParseExceptionMarker(ExceptionMarker):
    INCOMPLETE_WSVEC_FILE = 'The seedname_wsvec.dat file is empty or incomplete.'
    AMBIGUOUS_NEAREST_ATOM_POSITIONS = 'The nearest atom to use for position parsing is ambiguous.'  # pylint: disable=invalid-name
