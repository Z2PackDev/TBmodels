"""
Defines custom exceptions to be used in TBmodels. The main purpose of
these custom exceptions is their use in the CLI. Custom exceptions are
formatted in the CLI in a way that makes error parsing simple.
"""

from enum import Enum

import click

__all__ = (
    "TbmodelsException",
    "ExceptionMarker",
    "ParseExceptionMarker",
    "SymmetrizeExceptionMarker",
)


class ExceptionMarker(Enum):
    """Base class for markers to be used with :class:`.TbmodelsException` exceptions.

    .. note ::

        The exception marker **names** are considered a public interface and
        will not be changed in a minor release (although new ones
        can be added). The **values** of the exception markers however
        are informational only, and should not be relied upon.
    """


class ParseExceptionMarker(ExceptionMarker):
    """
    Exception markers for errors which can occur while parsing a
    tight-binding model.
    """

    INCOMPLETE_WSVEC_FILE = "The seedname_wsvec.dat file is empty or incomplete."
    AMBIGUOUS_NEAREST_ATOM_POSITIONS = "The nearest atom to use for position parsing is ambiguous."  # pylint: disable=invalid-name


class SymmetrizeExceptionMarker(ExceptionMarker):
    """
    Exception markers for errors which can occur while symmetrizing a
    tight-binding model.
    """

    INVALID_SYMMETRY_TYPE = "The type of the given symmetries object is incorrect."
    POSITIONS_NOT_SYMMETRIC = "The model positions do not respect the given symmetry."


class TbmodelsException(click.ClickException):
    """
    Custom exception class for TBmodels errors. This class should be
    used only for exception with a well-known cause, not for unexpected
    "crashes". For example, it can be used for malformed or incompatible
    inputs, truncated or missing input files, and similar errors.

    Errors which use this exception class will be formatted in the CLI as::

        Error: [<exception marker name>] <error message>
    """

    exit_code = 3

    def __init__(self, message: str, exception_marker: ExceptionMarker):
        super().__init__(message)
        self.exception_marker = exception_marker

    def format_message(self):
        return f"[{self.exception_marker.name}] {super().format_message()}"
