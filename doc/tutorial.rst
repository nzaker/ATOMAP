.. _tutorial:

========
Tutorial
========

Starting the interactive python environment in Ubuntu 16.04:

.. code-block:: bash

    $ ipython3

The first step is importing the module:

.. code-block:: python

    >>> import atomap

To analyse a single sublattice, use the
:py:meth:`~.atomap.atom_finding_refining.plot_feature_separation`.

.. code-block:: python

    >>> from atomap.atom_finding_refining import plot_feature_separation
    >>> import hyperspy.api as hs
    >>> s = hs.load("dataset.hdf5")
    >>> plot_feature_separation(s.data)

This functions find the most intense features in the image, with a
range of minimum distance between the features. This is plotted as
several images. Choose the pixel separation where there are no
extra markers between the atoms in the first sublattice, and all the
atoms in the first sublattice is covered by a marker.

Note, the dataset must be calibrated in nano meters.

.. code-block:: python

    >>> from atomap.main import make_atom_lattice_from_image
    >>> atom_lattice = make_atom_lattice_from_image("dataset.hdf5", pixel_separation=26)

Depending on the size of the dataset, this can take a while.

This returns an `atom_lattice` object, which contains several utility functions.
For example `plot_all_sub_lattices`, which plots all the atom column positions
on the image:

.. code-block:: python

    >>> atom_lattice.plot_all_sub_lattices()

This is saved as an image file ("all_sub_lattice.jpg").

Sublattices can be accessed using `atom_lattice.sub_lattice_list`:

.. code-block:: python

    >>> sub_lattice = atom_lattice.sub_lattice_list[0]

These `sublattice` objects contain a large amount of data and utility functions.
For example:

.. code-block:: python

    >>> sub_lattice.x_position
    >>> sub_lattice.y_position
    >>> sub_lattice.plot_monolayer_distance_map()

