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

    $ pip3 install --user atomap

`HyperSpy <http://hyperspy.org/>`_ is also included, due to Atomap relying heavily on the modelling functionality in HyperSpy.

To check that everything is working, open a terminal and run :code:`ipython3 --matplotlib qt4`. In the ipython terminal run:

.. code-block:: python

    import hyperspy.api as hs
    import atomap.api as am

If this works, continue with the :ref:`tutorial`.
If you get some kind of error, please report it as a New issue on the `Atomap GitLab <https://gitlab.com/atomap/atomap/issues>`_.

Windows
-------

WinPython HyperSpy installer
############################

Currently the easiest way to install Atomap is by using the WinPython HyperSpy installer.
Firstly download and install the `WinPython HyperSpy bundle <http://hyperspy.org/download.html#windows-bundle-installers>`_:
HyperSpy-1.3 for Windows 64-bits.

After installing the bundle, there should be a folder in the start menu called "HyperSpy WinPython Bundle", and this
folder should contain the "WinPython prompt". Start the "WinPython prompt". This will open a terminal window called
"WinPython prompt", in this window type and run:

.. code-block:: bash

    pip install atomap

To check everything is working correctly, go to the "HyperSpy WinPython Bundle" and start "Jupyter QtConsole".
This will open a new window. In this window, run:

.. code-block:: python

    %matplotlib qt4
    import hyperspy.api as hs
    import atomap.api as am

If this works, continue with the :ref:`tutorial`.
If you get some kind of error, please report it as a New issue on the `Atomap GitLab <https://gitlab.com/atomap/atomap/issues>`_.


Alternative Windows installation
################################

If you already have HyperSpy running in an Anaconda Python environment `Anaconda environment <https://www.continuum.io/downloads>`_,
Atomap can be installed from the `Anaconda prompt` (Start menu - Anaconda3), this will open a command line prompt.
In this prompt run:

.. code-block:: bash

    $ pip install atomap

