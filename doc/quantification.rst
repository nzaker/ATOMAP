.. _quantification:

==============
Quantification
==============

Atomap contains two methods for performing quantification of atomic-resolution electron microscopy images.

The methods termed "Absolute Integrator" integrate over Voronoi cells or watershedded regions and compare the intensity in this region to the total beam intensity.
When normalising for the detector response an accurate estimate of the percentage of the beam incident on the ADF detector for each atomic column can be obtained, which can be related to the scattering cross-section.

Atomap also contains an implementation of the "Statistical" method of ADF atomic-resolution quantification.
In this method, the intensity of a Gaussian fit to each atomic column is plotted as a histogram.
A Gaussian mixture modelo is then fit to this distribution, with each Gaussian corresponding to a different number of atoms in the columns.


Absolute Integrator
===================

The following section describes methods incorporated from the AbsoluteIntegrator code for normalisation and quantification of ADF STEM images.

.. For a full example please see the notebook in the Atomap-demos repository: https://gitlab.com/atomap/atomap_demos/adf_quantification

Detector Normalisation
----------------------

:py:func:`~atomap.quantification.detector_normalisation`

To carry out normal detector normalisation only the detector image and experimental image are needed.

.. code-block:: python

    >>> import hyperspy.api as hs
    >>> import atomap.api as am
    >>> det_image = am.example_data.get_detector_image_signal()
    >>> image = am.dummy_data.get_simple_cubic_signal(image_noise=True)
    >>> image_normalised = am.quant.detector_normalisation(image, det_image, 60)


Flux Weighting Analysis
-----------------------

In order to have a flux exponent to include in the detector normalisation (above), a flux analysis must be carried out.
The detector flux weighting method is based on the following paper:

(1) G.T. Martinez et al. Ultramicroscopy 2015, 159, 46-58.

.. code-block:: python

    >>> image_normalised = am.quant.detector_normalisation(image, det_image, inner_angle=60, outer_angle = None, flux_expo=2.873)


If the flux_exponent is unknown then it is possible to create an interactive flux plot described in detail in the example notebook: https://gitlab.com/atomap/atomap_demos/blob/release/adf_quantification_notebook/adf_quantification.ipynb


Statistical Method
==================

In order to perform the "statistical method" you must first have a ``Sublattice`` defined and you must also have used the Gaussian refinement.
The code below shows this process operating on an ADF image from the adf__quantification_notebook in the atomap demos.

.. code-block:: python

    >>> import atomap.api as am
    >>> data = am.dummy_data.get_atom_counting_data()
    >>> atom_positions = am.get_atom_positions(tdata, 8, threshold_rel=0.1)
    >>> sublattice = am.Sublattice(atom_positions, tdata.data)
    >>> sublattice.construct_zone_axes()
    >>> sublattice.refine_atom_positions_using_2d_gaussian()


Get number of Gaussians in GMM
------------------------------

Unless you already know the number of Gaussians to fit in your Gaussian mixture model, i.e. the number of different numbers of atoms in all of the columns, it is necessary to get this information.
In order to obtain this number, you can fit Gaussian mixture models with different numbers of Gaussians and judge the best model by an information criterion (here we plot both AIC and BIC).
Typically, a large negative gradient in the AIC/BIC is associated with the correct number of Gaussians, i.e. you're looking for a local minimum.

To obtain plots of AIC and BIC for your image, you use the :py:func:`~atomap.quantification.get_statistical_quant_criteria` function.

.. code-block:: python

    >>> models = am.quant.get_statistical_quant_criteria([sublattice], 20)

.. figure:: images/quant/criteria_plot.png
    :scale: 80 %

The :py:func:`~atomap.quantification.get_statistical_quant_criteria` function takes a list of sublattices as an argument so that you can obtain a model for an individual image or a collection of images.
For multiple images make sure that they were all acquired with the same beam current and detector settings.


Apply the selected model
------------------------

Once you have determined the number of Gaussians in your Gaussian mixture model, you can input this in to the :py:func:`~atomap.quantification.statistical_quant` function.

.. code-block:: python

    >>> model = models[12] # 13th model
    >>> statistical_quant(sublattice.image, sublattice, model, 13)

The function returns an ``Atom_Lattice`` object, in which each ``Sublattice`` corresponds to atomic columns of different atomic number.
If plotting is selected (as it is by default) this plots the histogram of column intensities with the Gaussian mixture model overlayed.
It also displays the image of the particle with sublattices coloured differently to indicate number of atoms in each column.

.. figure:: images/quant/quant_output.png
    :scale: 80 %
