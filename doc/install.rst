.. _install:

==========
Installing
==========

Linux
-----

The recommended way to install Atomap is using PIP, which is a package manager for python.
It is recommended to install the precompiled requirements using the system package manager.

To install the requirements in Ubuntu (16.10):

.. code-block:: bash

    $ sudo apt-get install python3-pip python3-numpy python3-scipy python3-h5py ipython3 python3-matplotlib python3-natsort python3-sklearn python3-dill python3-ipython-genutils python3-skimage
    $ sudo apt-get install python3-sympy --no-install-recommends

Installing Atomap itself:

.. code-block:: bash

    $ pip install --user atomap

`HyperSpy <http://hyperspy.org/>`_ is also included, due to Atomap relying heavily on the modelling functionality in HyperSpy.

Windows
-------

The easiest way to install is by using the `Anaconda environment <https://www.continuum.io/downloads>`_.
After installing the Python 3.5 version of Anaconda, open the *Anaconda prompt* (Start menu - Anaconda3).
This will open a command line prompt.

.. code-block:: bash

    $ conda install hyperspy
    $ conda install pyqt=4
    $ pip install atomap

Alternatively, if the HyperSpy Winpython bundle is installed Atomap can be installed from the *WinPython prompt*:

.. code-block:: bash

    $ pip install atomap

Getting started
---------------


A tutorial is located here: :ref:`tutorial`.

