.. (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
.. Author: Dominik Gresch <greschd@gmx.ch>

.. _reference:

Reference
=========

.. contents:: Contents
    :local:

Model Class
-----------

.. autoclass:: tbmodels.Model
    :members:
    :special-members: __add__, __mul__, __neg__, __rmul__, __sub__, __truediv__
    :inherited-members:


Helper functions
----------------

.. automodule:: tbmodels.helpers
    :members:
    :imported-members:

Saving and loading (HDF5)
-------------------------

.. automodule:: tbmodels.io
    :members:
    :imported-members:

.. _cli_reference:

Command line interface
----------------------

.. click:: tbmodels._cli:cli
    :prog: tbmodels
    :show-nested:


k.p Model class
---------------

.. autoclass:: tbmodels.kdotp.KdotpModel


Exceptions
----------

.. automodule:: tbmodels.exceptions
    :members:
    :undoc-members:
