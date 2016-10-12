.. _tutorial:

Tutorial
========

The following tutorial will guide you through the basic steps of using the TBmodels package: Creating, evaluating and saving a :class:`.Model` instance.

.. contents:: Contents
    :local:

Creating a Model instance
-------------------------

The :class:`.Model` class describes tight-binding models within TBmodels. Instances of this class can be created in various ways, some of which I will describe here.

First off, when creating a tight-binding model from first-principles using Wannier90, you get a file containing all the hopping terms. This file, called ``wannier90_hr.dat`` (or ``*_hr.dat``, where ``*`` is the seedname of Wannier90), can be read directly with TBmodels:

.. code:: python

    import tbmodels
    
    model = tbmodels.Model.from_hr_file('path_to_directory/wannier90_hr.dat')
   
Alternatively, a :class:`.Model` instance can be created directly using the constructor. The following example shows how to create a model with two orbitals. The orbitals have on-site energies ``1`` and ``-1`` (the unit can be arbitrary, but must be consistent), and there is one occupied state. The system is three-dimensional, and the orbitals are located at the origin and ``[0.5, 0.5, 0.]`` (in reduced coordinates), respectively.

In a second step, hopping terms between the two orbitals (nearest-neighbour interaction) and between the same orbital in different unit cells (next-nearest-neighbour) are added using the :meth:`.add_hop` method.
   
.. include:: simple_model.py
    :code: python
    
Evaluating the model
--------------------

Once created, a :class:`.Model` instance can be evaluated at different k-points in the Brillouin zone using the :meth:`.hamilton` and :meth:`.eigenval` methods. These methods take a single k-point, given in reduced coordinates, as argument.

.. code:: python

    print(model.hamilton(k=[0., 0., 0.]))
    print(model.eigenval(k=[0., 0., 0.]))


Saving the model to a file
--------------------------

There are different ways of saving the model to a file. To save the model for later use, I recommend using the :meth:`.to_json_file` method. This will preserve the model exactly as it is. 

.. code:: python
    
    model.to_json_file('model.json')
    model2 = tbmodels.Model.from_json_file('model.json') # model2 is an exact copy of model


If compatibility with other codes operating on Wannier90's ``*hr.dat`` format is needed, the :meth:`.to_hr_file` method can be used. However, this preserves only the hopping terms, not the positions of the atoms or shape of the unit cell. Also, the precision of the hopping terms is truncated.

.. code:: python

    model.to_hr_file('model_hr.dat')
    model3 = tbmodels.Model.from_hr_file('model_hr.dat') # model3 might differ from model

Finally, the :class:`.Model` class is also compatible with Python's built-in :py:mod:`pickle` module. However, data saved with :py:mod:`pickle` may not be readable with different versions of TBmodels since pickle serialization depends on the specific names of classes and their attributes. The use of :py:mod:`pickle` compatibility is to enable :py:mod:`multiprocessing` with TBmodels.
