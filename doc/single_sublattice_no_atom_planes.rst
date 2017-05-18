.. _single_sublattice_no_atom_planes:

===========================
Only finding atom positions
===========================

If you only want to find the atomic positions in a single sublattice
the make_atom_lattice_single_sublattice_from_image function can be used.
This function does not require the structure to be the same across the
whole image.

Using the same files and pixel_separation as in the :ref:`tutorial`

.. code-block:: python

    >>> mkdir atomap_testing1
    >>> cd atomap_testing1
    >>> import urllib.request
    >>> urllib.request.urlretrieve("https://gitlab.com/atomap/atomap/raw/master/atomap/tests/datasets/test_ADF_cropped.hdf5", "test_ADF_cropped.hdf5")
    >>> import hyperspy.api as hs
    >>> s = hs.load("test_ADF_cropped.hdf5")
    >>> import atomap.api as am
    >>> atom_lattice = am.make_atom_lattice_single_sublattice_from_image(s, pixel_separation=19)

Using this function, we only the get one sublattice and only the information
about the individual atom positions. Atom planes are not found, so the
functions which rely on these do not work.

So these work:

.. code-block:: python

    >>> sublattice = atom_lattice.sublattice_list[0]
    >>> sublattice.x_position
    >>> sublattice.y_position
    >>> sublattice.sigma_x
    >>> sublattice.sigma_y
    >>> sublattice.ellipticity
    >>> sublattice.rotation
    >>> sublattice.get_atom_list_on_image().plot(plot_markers=True)

While these do not work, since they rely on how the atom positions
relate to each other (distance, angle, ...):

.. code-block:: python

    >>> sublattice.get_monolayer_distance_map().plot(plot_markers=True)
    >>> sublattice.get_atom_distance_difference_map().plot(plot_markers=True)

