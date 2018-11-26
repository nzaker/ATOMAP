.. _gui_functions:

=================
GUI functionality
=================

.. _atom_adder_gui:

Adding atoms using GUI
======================

For most cases the majority of the atoms is found with automatic peak finding using :py:func:`~atomap.atom_finding_refining.get_feature_separation` and :py:func:`~atomap.atom_finding_refining.get_atom_positions`.
However, for some datasets there might be either missing or extra atoms.
These can be added or removed using :py:func:`~atomap.initial_position_finding.add_atoms_with_gui`.
This function opens up a window showing the datasets, where atoms can be added or removed by clicking on them with the mouse pointer.


.. code-block:: python

   >>> %matplotlib nbagg # doctest: +SKIP
   >>> s = am.dummy_data.get_distorted_cubic_signal()
   >>> atom_positions = am.get_atom_positions(s, 25)
   >>> atom_positions_new = am.add_atoms_with_gui(s, atom_positions)

.. image:: images/atomadderremovergui/atoms_add_remove_gui.gif
    :scale: 50 %
    :align: center


After having added or removed the atoms, ``atom_positions_new`` is used to make a sublattice object:

.. code-block:: python

   >>> sublattice = am.Sublattice(atom_positions_new, s)


:py:func:`~atomap.initial_position_finding.add_atoms_with_gui` can also be used without any initial atoms:


.. code-block:: python

   >>> atom_positions = am.add_atoms_with_gui(s)


If the atoms in the dataset are too close together, ``distance_threshold`` is used to decrease the distance for removing an atom.


.. code-block:: python

   >>> atom_positions = am.add_atoms_with_gui(s, distance_threshold=2)


.. _toggle_atom_refine:

Toggle atom refine
==================

To disable position refining or fitting of atoms in a sublattice, use :py:meth:`~atomap.sublattice.Sublattice.toggle_atom_refine_position_with_gui`:

.. code-block:: python

   >>> %matplotlib qt # doctest: +SKIP
   >>> sublattice = am.dummy_data.get_distorted_cubic_sublattice()
   >>> sublattice.toggle_atom_refine_position_with_gui()

Use the left mouse button to toggle refinement of the atom positions.
Green: refinement.
Red: not refinement.

.. image:: images/togglerefineposition/toggle_refine_position.gif
    :scale: 50 %
    :align: center

This can also be set directly through the `refine_position` property in the `Atom_Position` objects.

.. code-block:: python

   >>> sublattice.atom_list[5].refine_position = False
