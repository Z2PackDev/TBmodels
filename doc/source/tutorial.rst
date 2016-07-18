.. _tutorial:

Tutorial
========

The following tutorial will guide you through the basic steps of using the TBmodels package: Creating, evaluating and saving a :class:`.Model` instance.

.. contents:: Contents
    :local:

Creating a Model instance
-------------------------

.. code:: python

    import tbmodels
    
    model = tbmodels.Model.from_hr_file('wannier90_hr.dat')
    
Evaluating the model
--------------------

Once created, a :class:`Model` instance can be evaluated at different k-points in the Brillouin zone using the :meth:`.hamilton` and :meth:`.eigenval` methods. These methods take a single k-point, given in reduced coordinates, as argument.

.. code:: python

    print(model.hamilton(k=[0., 0., 0.]))
    print(model.eigenval(k=[0., 0., 0.]))


Saving the model to a file
--------------------------

There are different ways of saving the model to a file. To save the model for later use, I recommend using the :meth:`.to_json_file` method. This will preserve the model exactly as it is. 

.. code:: python
    
    model.to_json_file('model.json')


If compatibility with other codes operating on Wannier90's ``*hr.dat`` format is needed, the :meth:`.to_hr_file` method can be used.


.. code:: python

    model.to_hr_file('model_hr.dat')

Finally, the :class:`Model` class is also compatible with Python's built-in :py:mod:`pickle` module. However, data saved with :py:mod:`pickle` may not be readable with different versions of TBmodels since pickle serialization depends on the specific names of classes and their attributes. The use of :py:mod:`pickle` compatibility is to enable :py:mod:`multiprocessing` with TBmodels.
