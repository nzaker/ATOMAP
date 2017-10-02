.. _analysing_atom_lattices:

=======================
Analysing atom lattices
=======================

After finding and refining the atom lattices as shown in :ref:`finding_atom_lattices`, the atomic structure can be analysed through

1. Ellipticity of the atomic columns
2. Monolayer separation
3. Angle between monolayers
4. Intensity (coming soon)

In this tutorial we will use a dummy image containing two sublattices.
Different structural distortions have been introduced in the image, and this tutorial will study these distortions.
The plots that are made are simple plots that clearly shows the structural change of the sublattices.
It is also possible to make more fancy figures for publications (such as in the paper presenting `Atomap <https://dx.doi.org/10.1186/s40679-017-0042-5>`_), and some tips for doing this will be presented at the end of this tutorial.

Fantasite, the dummy structure
==============================

The procedure for finding and refining the sublattices in fantasite is similar as in :ref:`Images with more than one sublattice`.

.. code-block:: python

    >>> import atomap.api as am
    >>> from atomap.tools import remove_atoms_from_image_using_2d_gaussian
    
    >>> s = am.dummy_data.get_fantasite()
    >>> A_positions = am.get_atom_positions(s,separation=16)
    >>> sublattice_A = am.Sublattice(A_positions,image=s.data, color='r', name='A')
    >>> sublattice_A.find_nearest_neighbors()
    >>> sublattice_A.refine_atom_positions_using_center_of_mass()
    >>> sublattice_A.refine_atom_positions_using_2d_gaussian()
    >>> sublattice_A.construct_zone_axes()
    
    >>> direction_001 = sublattice_A.zones_axis_average_distances[1]
    >>> B_positions = sublattice_A._find_missing_atoms_from_zone_vector(direction_001)
    >>> image_without_A = remove_atoms_from_image_using_2d_gaussian(sublattice_A.image, sublattice_A)
    
    >>> sublattice_B = am.Sublattice(B_positions, image_without_A, color='blue', name='B')
    >>> sublattice_B.construct_zone_axes()
    >>> sublattice_B.refine_atom_positions_using_center_of_mass()
    >>> sublattice_B.refine_atom_positions_using_2d_gaussian()
    >>> atom_lattice = am.Atom_Lattice(image=s.data, name='fantasite', sublattice_list=[sublattice_A, sublattice_B])
    >>> atom_lattice.save("fantasite.hdf5", overwrite=True)
    >>> s.plot() 
    >>> atom_lattice.plot()

.. image:: images/plotting_tutorial/fantasite.png
    :scale: 50 %

.. image:: images/plotting_tutorial/atom_lattice.png
    :scale: 50 %

Fantasite is shown in the left image, and it is possible to see some variations in ellipticity and atom positions with the naked eye.
The right figure shown the atom positions after refinement.
As refinement of the sublattice can be time consuming, it is a good idea to save the final atom lattice.

The `Atom_Lattice` object 
=========================

The atom lattice can be loaded:

.. code-block:: python

    >>> import atomap.api as am
    >>> atom_lattice = am.load_atom_lattice_from_hdf5("fantasite.hdf5")
    >>> atom_lattice  # doctest: +SKIP
    <Atom_Lattice, fantasite (sublattice(s): 2)> # doctest: +SKIP
    >>> atom_lattice.sublattice_list  # doctest: +SKIP
    [<Sublattice,  (atoms:104,planes:6)>, <Sublattice,  (atoms:91,planes:6)>]  # doctest: +SKIP
    >>> image = atom_lattice.image0

:py:class:`atomap.atom_lattice.Atom_Lattice` is an object containing the sublattices, and other types of information.
The fantasite atom lattice contains two sublattices (red and blue dots in the image above).
Atom positions, sigma, ellipticity and rotation for the atomic columns in a sublattice can be accessed.

.. code-block:: python

    >>> sublattice_A = atom_lattice.sublattice_list[0]
    >>> x = sublattice_A.x_position
    >>> y = sublattice_A.y_position
    >>> sigma_x = sublattice_A.sigma_x
    >>> sigmal_y = sublattice_A.sigma_y
    >>> ellipticity = sublattice_A.ellipticity
    >>> rotation = sublattice_A.rotation_ellipticity

Similarly, properties of a single atomic column :py:class:`atomap.atom_position.Atom_Position` can be accessed though :py:attr:`atomap.sublattice.Sublattice.atom_list`.
The :py:class:`atomap.atom_position.Atom_Position` object contain information related to a specific atomic column.

.. code-block:: python

    >>> atom_position_list = sublattice_A.atom_list
    >>> atom_position = atom_position_list[0]
    >>> x = atom_position.pixel_x
    >>> y = atom_position.pixel_y
    >>> sigma_x = atom_position.sigma_x
    >>> sigma_y = atom_position.sigma_y

The :py:class:`atomap.atom_plane.Atom_Plane` objects contain the atomic columns belonging to the same specific plane.
Atom plane objects are defined by the direction vector parallel to the atoms in the plane, for example (58.81, -41.99).
These can be accessed by:

.. code-block:: python

    >>> atom_plane_list = sublattice_A.atom_plane_list
    >>> atom_plane = atom_plane_list[0]
    >>> atoms_in_plane_list = atom_plane.atom_list

Ellipticity
===========

Elliptical atomic columns may occur when atoms parallel to the electron beam have sifted position in the plane orthogonal to the beam.
In the image, circular atomic columns have an ellipticity (:math:`\epsilon`) of 1, as `sigma_x`  = `sigma_y` (:math:`\sigma_x = \sigma_y`).
Ellipticity is defined as

.. math::

    \epsilon = 
        \begin{cases}
                \frac{\sigma_x}{\sigma_y},& \text{if } \sigma_x > \sigma_y\\
                        \frac{\sigma_y}{\sigma_x},& \text{if } \sigma_y > \sigma_x\\
                            \end{cases}


Ellipticity maps
----------------
The ellipticity map shows the magnitude of the ellipticity.
Values are interpolated, giving a continuous map.
The sublattice B was generated without any ellipticity, and the image to the right is fairly flat, as expected.
In sublattice A, a region with elliptical atomic columns is clearly visible.
The ellipticity also increases from left to right towards a maximum, before it starts to fall again.
This is perfectly in line with how the dummy image of fantasite has been generated.
Maps gives nice visualization of gradual change.

.. code-block:: python

    >>> sublattice_A = atom_lattice.sublattice_list[0]
    >>> sublattice_B = atom_lattice.sublattice_list[1]
    >>> sublattice_A.plot_ellipticity_map(cmap='viridis',vmin=0.95,vmax=1.3)
    >>> sublattice_B.plot_ellipticity_map(cmap='viridis',vmin=0.95,vmax=1.3)

.. image:: images/plotting_tutorial/ellipticity_map_A.png
    :scale: 50 %
    
.. image:: images/plotting_tutorial/ellipticity_map_B.png
    :scale: 50 %

The :py:meth:`atomap.sublattice.Sublattice.plot_ellipticity_map` function calls :py:meth:`atomap.sublattice.Sublattice.get_ellipticity_map`, which calls :py:meth:`atomap.sublattice.Sublattice._get_property_map`. 

Vector plots
------------
While the ellipticity map nicely visualizes the magnitude (and gradual change) of the ellipticity, it does not show the direction of the ellipse.
In vector (quiver) plots (:py:meth:`atomap.sublattice.Sublattice.plot_ellipticity_vectors`) both the rotation and magnitude are visualized, through the length and angle of the arrows.
There is one arrow for each atom position.

.. code-block:: python

    >>> sublattice_A.plot_ellipticity_vectors()

.. image:: images/plotting_tutorial/ellipticity_vectors.png
    :align: center
    :scale: 50 %

In this function, a value of 1 is subtracted from the magnitude of the ellipticity.
This makes it easier to study changes in ellipticity, as the 0-point of the plot is set to the perfect circle.
:py:meth:`atomap.plotting.plot_vector_field` is called, and this function uses `Matplotlib's quiver plot function <https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.quiver.html?highlight=quiver#matplotlib.axes.Axes.quiver>`_.

Distance between monolayers
===========================

As Atomap knows the positions of all atoms, it can also tell you if you have strain or other types of structural distortions.
For example, Atomap has been used to study `oxygen octahedron tilt patterns in perovskite thin films <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.94.201115>`_. 

In this example, Atomap finds the distance between monolayers.
"Distance between monolayers" is defined in the below figure (a).
The distance is measured between an atomic column and the nearest monolayer, as shown in this figure.

.. image:: images/plotting_tutorial/definition.jpg
    :align: center
    :scale: 50 %

`s_monolayer` is a `HyperSpy signal stack <http://hyperspy.org/hyperspy-doc/current/user_guide/tools.html>`_, where the navigation axis is a zone vector (the values in the zone axis tuple are in principle the vector parallel to the monolayer) and the signal axes shows monolayer separation at each atom position.
:py:meth:`atomap.sublattice.Sublattice.get_monolayer_distance_map` can also take in a subset of zone vectors, but the default is to find the monolayer separation for all the zone axis.

.. code-block:: python

    >>> s_monolayer = sublattice_B.get_monolayer_distance_map()
    >>> s_monolayer.plot(cmap='viridis')


.. image:: images/plotting_tutorial/Sublattice_B_monolayer_distance_a.png
    :scale: 50 %

.. image:: images/plotting_tutorial/Sublattice_B_monolayer_distance_b.png
    :scale: 50 %


The left image shows the monolayer separation for one zone axis, namely the separation between the monolayers drawn up by red lines in the right figure.
Clearly, the position of the B atomic columns are changed in the middle of the image, where every second monolayer is closer and more far apart from the atom. 

Angle between atoms
===================

This example shows how the angle between atoms can be visualized.
:py:meth:`atomap.sublattice.Sublattice.get_atom_angles_from_zone_vector` is used, and this function returns three lists: x- and y- coordinates of the atoms, and a list of the angle
between two zone axies at each atom.

.. code-block:: python

    >>> z1 =  sublattice_B.zones_axis_average_distances[0]
    >>> z2 =  sublattice_B.zones_axis_average_distances[1]
    >>> x, y, a = sublattice_B.get_atom_angles_from_zone_vector(z1, z2, degrees=True)
    >>> s_angle = sublattice_B._get_property_map(x, y, a)
    >>> s_angle.plot()

.. image:: images/plotting_tutorial/Angle_map_z1.png
    :scale: 50 %

.. image:: images/plotting_tutorial/Angle_map_z2.png
    :scale: 50 %

.. image:: images/plotting_tutorial/Angle_map.png
    :scale: 50 %

.. image:: images/plotting_tutorial/Angle_map_zoom.png
    :scale: 50 %

Atomic columns start to "zigzag" in the rightmost part of the image.
This is also clear with the naked eye (atomic columns marked with blue dots). 
:py:meth:`atomap.sublattice.Sublattice._get_property_map` is a very general function, and can plot a map of any property.

Line profiles
=============

Often it can be a good idea to integrate parts of the image, for example to improve the signal-to-noise ratio or to reduce the information of fewer dimensions.
This example will introduce how line profiles of properties can be made, by projecting the property onto a specific atom plane.
The difference between mapping line profiles and maps, is that you need to decide the projection of the line profile.

Plotting for publication
========================

Please consider to cite `Atomap: a new software tool for the automated analysis of atomic resolution images using two-dimensional Gaussian fitting <https://ascimaging.springeropen.com/articles/10.1186/s40679-017-0042-5>`_ when you publish work where you have used Atomap as a tool.
Figures for publication are often customized, and here are a few tips on how to extract the data you wish to plot in a fancy plot.

Saving specific data
--------------------

When making advanced figures containing specific data for publication, it can be a good idea to save this data for example in separate numpy files.
This makes it quick to load the data when using for example matplotlib to make figures.

.. code-block:: python

    >>> import numpy as np
    >>> np.savez("datafile.npz", x=sublattice_A.x_position, y=sublattice_A.y_position)

Alternatively, the data can  ve saved in comma-separated values (CSV) file, which can be opened in spreadsheet software:

.. code-block:: python

    >>> np.savetxt("datafile.csv", (sublattice_A.x_position, sublattice_A.y_position, sublattice_A.sigma_x, sublattice_A.sigma_y, sublattice_A.ellipticity), delimiter=',')
    
 
Signals can be saved by using the inbuilt `save` function.

.. code-block:: python

    >>> s_monolayer.save("monolayer_distances.hdf5",overwrite=True)
