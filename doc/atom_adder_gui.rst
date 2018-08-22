.. _atom_adder_gui:

======================
Adding atoms using GUI
======================

For most cases the majority of the atoms is found with automatic peak finding using :py:func:`~atomap.atom_finding_refining.get_feature_separation` and :py:func:`~atomap.atom_finding_refining.get_atom_positions`.
However, for some datasets there might be either missing or extra atoms.
These can be added or removed using :py:func:`~atomap.initial_position_finding.add_atoms_with_gui`.
This function opens up a window showing the datasets, where atoms can be added or removed by clicking on them with the mouse pointer.


.. code-block:: python

   >>> %matplotlib qt # doctest: +SKIP
   >>> s = am.dummy_data.get_distorted_cubic_signal()
   >>> peaks = am.get_atom_positions(s, 25)
   >>> peaks_new = am.add_atoms_with_gui(s, peaks)

.. image:: images/atomadderremovergui/add_atoms.gif
    :scale: 50 %
    :align: center


After having added or removed the atoms, ``peaks_new`` is used to make a sublattice object:

.. code-block:: python

   >>> sublattice = am.Sublattice(peaks_new, s)


:py:func:`~atomap.initial_position_finding.add_atoms_with_gui` can also be used without any initial atoms:


.. code-block:: python

   >>> peaks = am.add_atoms_with_gui(s)


If the atoms in the dataset are too close together, ``distance_threshold`` is used to decrease the distance for removing an atom.


.. code-block:: python

   >>> peaks = am.add_atoms_with_gui(s, distance_threshold=2)

