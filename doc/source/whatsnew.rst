.. (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
.. Author: Dominik Gresch <greschd@gmx.ch>

.. _whatsnew:

What's new
==========

The following is a short summary of the most important changes in new releases of TBmodels, starting *after* version 1.1.

What's new in TBmodels 1.4 (development version)
------------------------------------------------

New features
''''''''''''

- Add ``--version`` option to the command-line interface, to print the current version.

- The ``KdotpModel`` has been promoted from an internal-only interface to the public ``tbmodels.kdotp`` submodule.

- The ``supercell`` method can be used to obtain a supercell tight-binding model.

- The ``hamilton`` and ``eigenval`` methods now also accept a list of k-points as input, returning a sequence of results in that case. This improves performance when evaluating many k-points.

- Add ``remove_small_hop`` and ``remove_long_range_hop`` methods to cut hoppings with small value or long distance, respectively. Both these methods operate in-place on the existing model.

Experimental features
'''''''''''''''''''''

- The ``fold_model`` method creates a tight-binding model for a smaller unit cell from a supercell model.

Deprecations and removals
'''''''''''''''''''''''''

- The ``from_hr`` and ``from_hr_file`` methods, deprecated since version 1.1, have been removed.


What's new in TBmodels 1.3
--------------------------

New features
''''''''''''

- **Constructing k.p models**: The :meth:`.construct_kdotp` method can be used to construct a k.p model from an existing tight-binding model.

- **Joining models**: The :meth:`.join_models` method can be used to combine models of the same structure with different orbitals.

Other improvements
''''''''''''''''''

- **Efficiency**: Empty hopping matrices are now automatically removed when constructing a new model. In some cases this can lead to significant performance improvements. No action is required to get these run-time benefits. To also reduce the size of existing models, they need to be loaded and saved.

Other changes
'''''''''''''

- **License**: TBmodels is now released under the more permissive Apache license.

What's new in TBmodels 1.2
--------------------------

New features
''''''''''''

- **Symmetrization feature**: The :meth:`.symmetrize` method allows symmetrizing tight-binding models by computing a group average, as described in `this paper <https://doi.org/10.1103/PhysRevMaterials.2.103805>`_.

- **Command-line interface**: A command-line interface for performing the most common tasks such as parsing models, symmetrizing, or evaluating bands was added.

- **Convenience function for Wannier90 parsing**: The :meth:`.from_wannier_folder` method was added to conveniently parse the output of a Wannier90 calculation, without having to specify each individual file.

Other changes
'''''''''''''

- **Update HDF5 file format**: The HDF5 file format was changed to be compatible with other tools in the Z2Pack ecosystem, such as ``symmetry-repesentation``. The old format is still supported, so no action is required.
