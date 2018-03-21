.. _absolute_integrator:

===================
Absolute Integrator
===================

The following section describes methods incorporated from the AbsoluteIntegrator code for normalisation and quantification of ADF STEM images.
For a full example please see the notebook in the Atomap-demos respository: https://gitlab.com/atomap/atomap_demos/adf_quantification

Detector Normalisation
======================

To carry out normal detector normalisation only the detector image and experimental image are needed.

>>> import hyperspy.api as hs
>>> import atomap.api as am
>>> det_image = am.example_data.get_detector_image_signal()
>>> image = am.dummy_data.get_simple_cubic_signal(image_noise=True)
>>> image_normalised = am.quant.detector_normalisation(image, det_image, 60)

Flux Weighting Analysis
=======================

In order to have a flux exponent to include in the detector normalisation (above), a flux analysis must be carried out.
The detector flux weighting method is based on the following paper:

(1) G.T. Martinez et al. Ultramicroscopy 2015, 159, 46-58.

..code-block:: python

>>> image_normalised = am.quant.detector_normalisation(image, det_image, inner_angle=60, outer_angle = None, flux_expo=2.873)

If the flux_exponent is unknown then it is possible to create an interactive flux plot described in detail in the example notebook: Atomap-demos respository: https://gitlab.com/atomap/atomap_demos/adf_quantification.
