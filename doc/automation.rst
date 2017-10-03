.. _automation:

============================
Automatation of the analysis
============================

If you are about to study many atomic resolution images of the same type of structure, you can save time and effort by using tools for automatation in Atomap by setting process parameters.
:py:mod:`atomap.process_parameters` is a module offering classes of predefined process parameters for some types of structures:

1. Perovskite Oxides (110-oriented), for all sublattices (A, B and Oxygen)
2. A Generic strucutre

This tutorial will first show how the use of these process parameters make the procedure for finding the full atom lattice more automatic.
Currently, predefined process parameters are available for a limited number of materials and projections.
The last part of this tutorial aims to show how such parameters can be made for new materials.

Finding atom lattices with process parameters
---------------------------------------------

In this tutorial we will use the predefined process parameter `PerovskiteOxide110`.
Tt contains various parameters and names for processing a perovskite oxide structure projected along the [110] direction.
The master function :py:meth:`atomap.main.make_atom_lattice_from_image` takes the atomic resolution signal, process parameters and optimal feature separation.
This means that you probably need to run :py:meth:`atomap.atom_finding_refining.get_feature_separation` and find the best pixel separation first.

.. code-block:: python

    >>> import atomap.api as am
    >>> process_parameter = am.process_parameters.PerovskiteOxide110()
    >>> atom_lattice = am.make_atom_lattice_from_image(s, process_parameter=process_parameter, pixel_separation=16) # doctest: +SKIP

Depending on the size of the dataset, this can take a while. 
The processing will:
    1. Locate the most intense atomic columns (A-cations, Strontium).
    2. Refine the position using center of mass.
    3. Refine the position using 2-D Gaussian distributions
    4. Find the translation symmetry using nearest neighbor statistics, and construct atomic planes using this symmetry.
    5. Locate the second most intense atomic columns (B-cation, Titanium), using the parameters defined in the model parameters
    6. "Subtract" the intensity of the A-cations from the HAADF image
    7. Refine the position of the B-cations using center of mass
    8. Refine the position of the B-cations using 2-D Gaussian distributions
    9. Construct atomic planes in the same way as for the first sublattice.

This returns the `atom_lattice` object, which contains the sublattices of both A and B cations.

.. code-block:: python

    >>> atom_lattice.plot() # doctest: +SKIP

.. image:: images/tutorial/atomlattice_plot_atoms.jpg
    :scale: 50 %
    :align: center
    
Making process parameters
-------------------------

This is how you make it.
Please consider to contribute to Atomap with you new process parameter class.
