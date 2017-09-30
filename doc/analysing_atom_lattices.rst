.. _analysing_atom_lattices:

=======================
Analysing atom lattices
=======================

After finding and refining the atom lattices as shown in ref:`finding_atom_lattices` , the atomic structure can be analysed through

1. The distance between neighbouring atoms
2. Monolayer separation
3. Ellipticity of the atomic columns
4. (Intensity)


.. code-block:: python

    >>> import atomap.api as am
    >>> signal = am.get_simple_cubic_signal()

.. image:: images/testdata/testdata_simple_cubic.png
    :scale: 50 %
    :align: center

The sublattice of the simple cubic structure can also be generated

.. code-block:: python

    >>> sublattice = am.get_simple_cubic_sublattice()
    <Sublattice,  (atoms:225,planes:0)>
    
More advanced datasets can also be created.
The functionality for generating testdata signals uses the function sublattice.get_model_image.
