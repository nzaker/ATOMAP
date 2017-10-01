.. _analysing_atom_lattices:

=======================
Analysing atom lattices
=======================

After finding and refining the atom lattices as shown in ref:`finding_atom_lattices` , the atomic structure can be analysed through

1. The distance between neighbouring atoms
2. Monolayer separation
3. Ellipticity of the atomic columns
4. (Intensity)

Load atom lattice
===================

.. code-block:: python

    >>> import atomap.api as am
    >>> atom_lattice = am.load_atom_lattice_from_hdf5("test_ADF_cropped_atom_lattice.hdf5") # doctest: +SKIP

Introduce the `atom_lattice` object, and how information is accessed.

.. code-block:: python

    >>> x = sublattice.x_position # doctest: +SKIP
    >>> y = sublattice.y_position # doctest: +SKIP
    >>> sigma_x = sublattice.sigma_x # doctest: +SKIP
    >>> sigmal_y = sublattice.sigma_y # doctest: +SKIP
    >>> ellipticity = sublattice.ellipticity # doctest: +SKIP
    >>> rotation = sublattice.rotation # doctest: +SKIP

These can be saved in different formats such as Numpy npz file:

.. code-block:: python

    >>> import numpy as np
    >>> np.savez("datafile.npz", x=sublattice.x_position, y=sublattice.y_position) # doctest: +SKIP

Or comma-separated values (CSV) file, which can be opened in spreadsheet software:

.. code-block:: python

    >>> np.savetxt("datafile.csv", (sublattice.x_position, sublattice.y_position, sublattice.sigma_x, sublattice.sigma_y, sublattice.ellipticity), delimiter=',')

Visualize structural properties
===============================

`sublattice` objects also contain a several plotting functions.
Since the image is from a |SrTiO3| single crystal, there should be no variations in the structure.
So any variations are due to factors such as scanning noise, sample drift and possibly bad fitting.

.. code-block:: python

    >>> s_monolayer = sublattice.get_monolayer_distance_map() # doctest: +SKIP
    >>> s_monolayer.plot() # doctest: +SKIP
    >>> s_elli = sublattice.get_ellipticity_map() # doctest: +SKIP
    >>> s_elli.plot() # doctest: +SKIP

These signals can be saved by using the inbuilt `save` function in the signals.

.. code-block:: python

    >>> s_monolayer.save("monolayer_distances.hdf5",overwrite=True) # doctest: +SKIP

The `sublattice` objects also contain a list of all the atomic planes:

.. code-block:: python

    >>> atom_plane_list = sublattice.atom_plane_list # doctest: +SKIP

The `atom_plane` objects contain the atomic columns belonging to the same specific plane.
Atom plane objects are defined by the direction vector parallel to the atoms in the plane, for example (58.81, -41.99).
These can be accessed by:

.. code-block:: python

    >>> atom_plane = atom_plane_list[0] # doctest: +SKIP
    >>> atom_list = atom_plane.atom_list # doctest: +SKIP
    
The atom planes can be plotted by using the `get_all_atom_planes_by_zone_vector` function, where the zone vector is changed by using the left-right arrow keys:

.. code-block:: python

    >>> sublattice.get_all_atom_planes_by_zone_vector().plot() # doctest: +SKIP

.. image:: images/tutorial/atomic_planes.jpg
    :scale: 50 %
    :align: center

The `atom_position` objects contain information related to a specific atomic column.
For example:

.. code-block:: python

    >>> atom_position = sublattice.atom_list[0] # doctest: +SKIP
    >>> x = atom_position.pixel_x # doctest: +SKIP
    >>> y = atom_position.pixel_y # doctest: +SKIP
    >>> sigma_x = atom_position.sigma_x # doctest: +SKIP
    >>> sigma_y = atom_position.sigma_y # doctest: +SKIP
    >>> sublattice.plot() # doctest: +SKIP

Basic information about the `atom_lattice`, `sublattice`, `atom_plane` and `atom_position` objects can be accessed by simply:

.. code-block:: python

    >>> atom_lattice # doctest: +SKIP
    <Atom_Lattice, signal (sublattice(s): 2)>
    >>> sublattice # doctest: +SKIP
    <Sublattice, A-cation (atoms:238,planes:6)>
    >>> atom_plane # doctest: +SKIP
    <Atom_Plane, (-0.19, -29.5) (atoms:17)>
    >>> atom_position # doctest: +SKIP
    <Atom_Position,  (x:322.4,y:498.8,sx:4.4,sy:5.1,r:1.3,e:1.2)>
