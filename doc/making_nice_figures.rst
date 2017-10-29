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

Om hva som skal plottes

.. literalinclude:: images/plot_for_pub.py


.. code-block:: python

    >>> import atomap.api as am
    >>> import hyperspy.api as hs
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import matplotlib.gridspec as gridspec
    >>> from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    >>> import matplotlib.font_manager as fm
    >>> import matplotlib.patheffects as patheffects

Om hva som importeres

.. code-block:: python

    >>> atom_lattice = am.dummy_data.get_fantasite_atom_lattice()
    
Om hva som er bilder

.. code-block:: python

    >>> im = atom_lattice.image0
    >>> s_adf = hs.signals.Signal2D(im)

Making the figure

.. code-block:: python

    >>> fig = plt.figure(figsize=(4.3, 2)) # in inches
    >>> gs = gridspec.GridSpec(1, 5)

    >>> ax_adf = plt.subplot(gs[:2])
    >>> ax_al = plt.subplot(gs[2:4])
    >>> ax_lp = plt.subplot(gs[4])

Making data

.. code-block:: python

    >>> sublattice = atom_lattice.sublattice_list[0]
    >>> sublattice.construct_zone_axes()
    >>> zone = sublattice.zones_axis_average_distances[0]
    >>> s_dd = sublattice.get_atom_distance_difference_map([zone])
    >>> s_dd = s_dd.isig[40.:460.,40.:460.]
    >>> s_adf = s_adf.isig[40.:460.,40.:460.]

Plot image data
    
.. code-block:: python

    >>> cax_adf = ax_adf.imshow(
                    np.rot90(s_adf.data),
            	    interpolation='nearest',
                    origin='upper',
            	    extent=s_adf.axes_manager.signal_extent)

Add scalebar

.. code-block:: python

    >>> fontprops = fm.FontProperties(size=12)
    >>> scalebar0 = AnchoredSizeBar(
        ax_adf.transData,
        10, '10 nm', 4,
        pad=0.1,
        color='white',
        frameon=False,
        size_vertical=1,
        fontproperties=fontprops)
        ax_adf.add_artist(scalebar0)

Add markers for sublattice

.. code-block:: python

    >>> atoms_A = atom_lattice.sublattice_list[0]
    >>> for idx, x in enumerate(atoms_A.x_position):
    >>>     y = atoms_A.y_position[idx]
    >>>     if (240 < x < 350) and (96 < y < 200):
    >>>         ax_adf.scatter(y, x, color='r', s=0.5)

    >>> atoms_B = atom_lattice.sublattice_list[1]
    >>> for idx, x in enumerate(atoms_B.x_position):
    >>>     y = atoms_B.y_position[idx]
    >>>     if (240 < x < 350) and (96 < y < 200):
    >>>         ax_adf.scatter(y, x, color='b', s=0.5)


Plot atom lattice property

.. code-block:: python

    >>> cax_al = ax_al.imshow(
                        np.rot90(s_dd.data),
                        interpolation='nearest',
                        origin='upper',
                        extent=s_adf.axes_manager.signal_extent,
                        cmap='viridis'
                        )

Scalebar

.. code-block:: python

    >>> scalebar1 = AnchoredSizeBar(
                ax_al.transData,
                10, '10 nm', 4,
                pad=0.1,
                color='white',
                frameon=False,
                size_vertical=1,
                fontproperties=fontprops)
    >>> ax_al.add_artist(scalebar1)

Remove ticks

.. code-block:: python

    >>> for ax in [ax_adf, ax_al]:
    >>> ax.set_xticks([])
    >>> ax.set_yticks([])


Plot line profile

.. code-block:: python

    >>> ax_lp.plot(s_dd_line.data, s_dd_line.axes_manager[0].axis)
    >>> ax_lp.set_xlabel("Distance difference", fontsize=8)
    >>> ax_lp.set_ylabel(r"Distance from interface", fontsize=8)
    >>> ax_lp.tick_params(axis='both', which='major', labelsize=6)
    >>> ax_lp.tick_params(axis='both', which='minor', labelsize=6)
    >>> ax_lp.yaxis.set_label_position('right')
    >>> ax_lp.yaxis.set_ticks_position('right')
    >>> ax_lp.set_ylim(-103, 317)

Add figure labels

.. code-block:: python

    >>> path_effects = [patheffects.withStroke(linewidth=2, foreground='black', capstyle="round")]
    >>> ax_adf.text(
            0.015,0.90,"a",fontsize=12, color='white',
            path_effects=path_effects,
            transform=ax_adf.transAxes)
    >>> ax_al.text(
            0.015,0.90,"b",fontsize=12, color='white',
            path_effects=path_effects,
            transform=ax_al.transAxes)
    >>> ax_lp.text(
            0.05,0.90,"c",fontsize=12, color='w',
            path_effects=path_effects,
            transform=ax_lp.transAxes)


Make margins and save

.. code-block:: python

    >>> gs.update(left=0.01, wspace=0.05, top=0.95, bottom=0.2, right=0.89)
    >>> plt.savefig('Atom_lattice.png', dpi=300)
    
.. image:: images/Atom_lattice.png

