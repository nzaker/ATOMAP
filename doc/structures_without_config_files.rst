.. _structures_without_config_files:

===============================
Structures without config files
===============================

Currently Atomap requires:

1. The image must be calibrated in nanometers
2. Atomic columns must be clearly resolved
3. The atomic structure must be similar across the whole image. If it's not the easiest solution is simply cropping the image

For an example on how to do this, see the no_config_parameters demo
script at the `atomap-demos
<https://gitlab.com/atomap/atomap_demos/tree/master/>`_
repository, which can be downloaded as a `zip file
<https://gitlab.com/atomap/atomap_demos/repository/archive.zip?ref=master>`_.

After unzipping, the example is found in no_config_parameters/no_config_parameters.py

The script can be used by navigating to the folder with a IPython terminal, then:

.. code-block:: python

    >>> run no_config_parameters.py # doctest: +SKIP

A more advanced example is found in no_config_parameters_two_sublattices.py,
which fits two separate sublattices in the same image.
