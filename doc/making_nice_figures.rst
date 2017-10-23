.. _making_nice_figures:

========================
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
    >>> np.savez("datafile.npz", x=sublattice_A.x_position, y=sublattice_A.y_position, e=sublattice_A.ellipticity) # doctest: +SKIP

Alternatively, the data can be saved in comma-separated values (CSV) file, which can be opened in spreadsheet software:

.. code-block:: python

    >>> np.savetxt("datafile.csv", (sublattice_A.x_position, sublattice_A.y_position, sublattice_A.sigma_x, sublattice_A.sigma_y, sublattice_A.ellipticity), delimiter=',') # doctest: +SKIP


Signals can be saved by using the inbuilt `save` function.

.. code-block:: python

    >>> s_monolayer.save("monolayer_distances.hdf5",overwrite=True) # doctest: +SKIP

Matplotlib
----------

Matplotib pyplot is our favourite plotting tool: gridspec,
