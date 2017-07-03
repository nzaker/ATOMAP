Welcome to Atomap's documentation!
==================================

Atomap is a Python library for analysing atomic resolution
scanning transmission electron microscopy images.
It relies in fitting 2-D Gaussian functions to every atomic
column in an image, and automatically find all major symmetry
axes. The full procedure is explained in the article 
`Atomap: a new software tool for the automated analysis of
atomic resolution images using two-dimensional
Gaussian fitting <https://dx.doi.org/10.1186/s40679-017-0042-5>`_.

.. figure:: images/index/elli_figure.jpg
    :scale: 45 %
    :align: center
    :target: https://ascimaging.springeropen.com/articles/10.1186/s40679-017-0042-5#Fig6

    Measuring the ellipticity of atomic columns. `More info <https://ascimaging.springeropen.com/articles/10.1186/s40679-017-0042-5#Fig6>`_


Instructions on how to install Atomap are found in the :ref:`install` guide.

There is a :ref:`tutorial` on how to use Atomap.

The source code is found in the `GitLab repository <https://gitlab.com/atomap/atomap/tree/master/>`_.

Atomap is under development and is in alpha, so bugs and errors can be expected.
Bug reports and feature requests are welcome on the `issue tracker <https://gitlab.com/atomap/atomap/issues>`_.

Atomap is available under the GNU GPL v3 license.

If you publish scientific articles using Atomap, please consider citing the article `Atomap: a new software tool for the automated analysis of
atomic resolution images using two-dimensional Gaussian fitting <https://dx.doi.org/10.1186/s40679-017-0042-5>`_.

.. figure:: images/index/oxygen_superstructure_figure.jpg
    :scale: 50 %
    :align: center
    :target: https://ascimaging.springeropen.com/articles/10.1186/s40679-017-0042-5#Fig5

    Mapping the variation in distance between oxygen columns. `More information <https://ascimaging.springeropen.com/articles/10.1186/s40679-017-0042-5#Fig5>`_

News
----
2017-07-03: version 0.0.8 released! 
New features: ability to process dumbbell structures, fitting of multiple 2D Gaussians at the same time, improved background subtraction during 2D Gaussian fitting, and processing of nanoparticles.

Contents on this webpage
------------------------

.. toctree::
   :maxdepth: 2

   install
   tutorial
   single_sublattice_no_atom_planes
   structures_without_config_files
   nanoparticle_example
   api_documentation


* :ref:`genindex`
* :ref:`modindex`

