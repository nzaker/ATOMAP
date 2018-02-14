.. _absolute_integrator:

===================
Absolute Integrator
===================

The following section describes methods incorporated from the AbsoluteIntegrator code for normalisation and quantification of ADF STEM images.
For a full example please see the notebook in the Atomap-demos respository: https://gitlab.com/atomap/atomap_demos/adf_quantification

Detector Normalisation
======================

To carry out normal detector normalisation only the detector image and experimental image are needed.

>>> import atomap.api as am
>>> det_image = hs.load()
>>> image = hs.load()
>>> image_normalised = am.quant.detector_normalisation(image,
                                                  det_image,
                                                  60)

Flux Weighting Analysis
=======================

In order to have a flux exponent to include in the detector normalisation (above), a flux analysis must be carried out.
The detector flux weighting method is based on the following paper:

(1) G.T. Martinez et al. Ultramicroscopy 2015, 159, 46–58.

Begin by creating an interacting flux plot using the following section of code:

..code-block:: python

  >>> flux_pattern = hs.load()
  >>> profiler = am.quant.find_flux_limits(flux_pattern.data, conv_angle=25)

This produces a profile from the flux_pattern image. On the resulting image, select the upper and lower limits for the flux analysis with left and right mouse clicks.
Use the central mouse button to confirm when you're happy with the selection.

The following section of code shows how to analyse the electron flux within the limits already selected by fitting an exponential curve to it.

..code-block:: python

  >>> coords = profiler[0].coords
  >>> flux_profile = profiler[1]
  >>> expo = am.quant.analyse_flux(coords, flux_profile, 25)

This can then be included into detector normalisation as follows:

..code-block:: python

  >>> image_normalised = am.quant.detector_normalisation (image,
                                                      det_img,
                                                      inner_angle=60,
                                                      outer_angle = 'None',
                                                      flux_expo=expo[0])
