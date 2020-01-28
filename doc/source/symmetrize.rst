.. (c) 2015-2018, ETH Zurich, Institut fuer Theoretische Physik
.. Author: Dominik Gresch <greschd@gmx.ch>

.. _symmetrize:

Symmetrization
==============

In this short tutorial, we introduce the use of the symmetrization feature in TBmodels. The theoretical background for this feature is described in `this paper <https://link.aps.org/doi/10.1103/PhysRevMaterials.2.103805>`_ (`arxiv version <https://arxiv.org/abs/1805.12148>`_). The example we describe here is to symmetrize a model of diamond silicon. The initial model, and code for this example, can also be found in the TBmodels source on `GitHub <https://github.com/Z2PackDev/TBmodels/tree/dev/examples/symmetrization/nonsymmorphic_Si>`_.


We start by loading the initial model:

.. ipython::

    In [0]: import tbmodels

The model is located in the ``examples`` directory of the TBmodels source. You will have to change the following part to point to your examples directory:

.. ipython::

    In [0]: import os
       ...: import pathlib
       ...: EXAMPLES_DIR = pathlib.Path(os.path.dirname(tbmodels.__file__)).parent / 'examples'

.. ipython::

    In [0]: model_nosym =  tbmodels.io.load(
       ...:     EXAMPLES_DIR / 'symmetrization' / 'nonsymmorphic_Si'/'data' / 'model_nosym.hdf5'
       ...: )


Setting up orbitals
-------------------

The most difficult step in using the symmetrization feature is setting up the symmetry operations. These need to be in the format of the  :mod:`symmetry_representation` module, which is also part of the Z2Pack ecosystem.

Starting with the 0.2 release, ``symmetry-representation`` contains some handy helper functions which allow us to construct the symmetry operations automatically by specifying the orbitals which constitute the tight-binding model. Thus, the first step to obtain the symmetry operations is to define the orbitals.

We can see that our initial model has two positions, which we store in the ``coords`` variable:

.. ipython::

    In [0]: model_nosym.pos

    In [0]: coords = ((0.5, 0.5, 0.5), (0.75, 0.75, 0.75))

Since the initial model was calculated with Wannier90 and ``sp3`` projections, we know that the orbitals are ordered as follows:

- spin up

  - first coordinate :math:`(0.5, 0.5, 0.5)`

    - :math:`sp^3` orbitals

  - second coordinate :math:`(0.75, 0.75, 0.75)`

    - :math:`sp^3` orbitals

- spin down

  - first coordinate :math:`(0.5, 0.5, 0.5)`

    - :math:`sp^3` orbitals

  - second coordinate :math:`(0.75, 0.75, 0.75)`

    - :math:`sp^3` orbitals

Consequently, we construct a list of :py:class:`symmetry_representation.Orbital` orbitals in this order:

.. ipython::

    In [0]: import symmetry_representation as sr

    In [0]: orbitals = []

    In [0]: for spin in (sr.SPIN_UP, sr.SPIN_DOWN):
       ...:     for pos in coords:
       ...:         for fct in sr.WANNIER_ORBITALS['sp3']:
       ...:             orbitals.append(sr.Orbital(
       ...:                 position=pos,
       ...:                 function_string=fct,
       ...:                 spin=spin
       ...:             ))

Here we used constants defined by ``symmetry_representation`` to specify the spin up / down components, and the :math:`sp^3` orbitals in the order produced by Wannier90.

.. ipython::

    In [0]: sr.SPIN_UP, sr.SPIN_DOWN

    In [0]: sr.WANNIER_ORBITALS['sp3']

The ``function_string`` argument is a string which describes the orbital in terms of the cartesian coordinates ``x``, ``y`` and ``z``. The ``symmetry-representation`` code will use ``sympy`` to apply the symmetry operations to these functions and figure out which orbitals these are mapped to.

Creating symmetry operations
----------------------------

Having created the orbitals which describe our system, we can immediately generate the symmetry operation for time-reversal symmetry:

.. ipython::

    In [0]: time_reversal = sr.get_time_reversal(orbitals=orbitals, numeric=True)

Note that we use the ``numeric=True`` flag here. This keyword is used to switch between output using ``numpy`` arrays with numeric content, and ``sympy`` matrices with analytic content. Mixing these two formats is a bad idea, since basic operations between them don't work as one might expect. For the use in TBmodels, we can **always** choose the ``numeric=True`` option.

Next, we use ``pymatgen`` to determine the space group symmetries of our crystal:

.. ipython::

    In [0]: import pymatgen as mg

    In [0]: structure = mg.Structure(
       ...:     lattice=model_nosym.uc, species=['Si', 'Si'], coords=np.array(coords)
       ...: )

    In [0]: analyzer = mg.symmetry.analyzer.SpacegroupAnalyzer(structure)

    In [0]: sym_ops = analyzer.get_symmetry_operations(cartesian=False)

    In [0]: sym_ops_cart = analyzer.get_symmetry_operations(cartesian=True)

Again, we can use a helper function from the ``symmetry-representation`` code to construct the symmetry operations automatically. Note that we need both the cartesian *and* the reduced symmetry operations:

.. ipython::

    In [0]: symmetries = []

    In [0]: for sym, sym_cart in zip(sym_ops, sym_ops_cart):
       ...:     symmetries.append(sr.SymmetryOperation.from_orbitals(
       ...:         orbitals=orbitals,
       ...:         real_space_operator=sr.RealSpaceOperator.from_pymatgen(sym),
       ...:         rotation_matrix_cartesian=sym_cart.rotation_matrix,
       ...:         numeric=True
       ...:     ))

Applying the symmetries
-----------------------

Finally, the simple task of applying the symmetries to the initial tight-binding model remains. We first apply the time-reversal symmetry.

.. ipython::

    In [0]: model_tr = model_nosym.symmetrize([time_reversal])

Note that, unlike the space group symmetries, the time-reversal symmetry does not constitute a full group. As a result, TBmodels will apply not only time-reversal :math:`\mathcal{T}`, but also :math:`\mathcal{T}^2 = -\mathbb{1}`, :math:`\mathcal{T}^3=-\mathcal{T}`, and the identity. For the space group, this extra effort is not needed since we already have the full group. This can be specified with the ``full_group=True`` flag:

.. ipython::

    In [0]: model_sym = model_tr.symmetrize(symmetries, full_group=True)

By comparing eigenvalues, we can see for example that the symmetrized model is two-fold degenerate at the :math:`\Gamma` point, while the initial model is not:

.. ipython::

    In [0]: model_nosym.eigenval((0, 0, 0))

    In [0]: model_sym.eigenval((0, 0, 0))
