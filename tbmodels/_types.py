#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Helper module defining custom types and protocols."""

__all__ = ('EqualComparable', )

import sys

if sys.version_info >= (3, 8):
    from typing import Protocol  # pylint: disable=no-name-in-module
else:
    from typing_extensions import Protocol


class EqualComparable(Protocol):
    def __eq__(self, other) -> bool:
        pass
