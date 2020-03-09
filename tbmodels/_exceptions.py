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
    """
    Custom exception class for TBmodels errors. Errors which use this
    exception class will be caught and formatted properly in the CLI.
    """
    #: The exit code for this exception
    exit_code = 3
    """Base class for the custom TBmodels exceptions."""
    def __init__(self, message, exception_marker: ExceptionMarker):
        super().__init__(message)
        self.exception_marker = exception_marker

    def format_message(self):
        return f'[{self.exception_marker.name}] {super().format_message()}'


class ParseExceptionMarker(ExceptionMarker):
    INCOMPLETE_WSVEC_FILE = 'The seedname_wsvec.dat file is empty or incomplete.'
    AMBIGUOUS_NEAREST_ATOM_POSITIONS = 'The nearest atom to use for position parsing is ambiguous.'  # pylint: disable=invalid-name
