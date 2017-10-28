.. _install:

==========
Installing
==========


Installing in Linux
-------------------

The recommended way to install Atomap is using PIP, which is a package manager for python.
It is recommended to install the precompiled requirements using the system package manager.

`HyperSpy <http://hyperspy.org/>`_ is also included, due to Atomap relying heavily on the modelling functionality in HyperSpy.

Ubuntu 17.10
************

.. code-block:: bash

    $ sudo apt-get install ipython3 python3-pip python3-numpy python3-scipy python3-matplotlib python3-sklearn python3-skimage python3-h5py python3-dask python3-traits python3-tqdm python3-pint python3-dask python3-pyqt4 python3-lxml
    $ sudo apt-get install python3-sympy --no-install-recommends
    $ pip3 install --user atomap


Ubuntu 16.04
************

.. code-block:: bash

    $ sudo apt-get install python3-pip python3-numpy python3-scipy python3-h5py ipython3 python3-natsort python3-sklearn python3-dill python3-ipython-genutils python3-skimage
    $ sudo apt-get install python3-sympy --no-install-recommends
    $ pip3 install --user hyperspy[all]
    $ pip3 install --user atomap

Starting Atomap
---------------

To check that everything is working, open a terminal and run :code:`ipython3 --matplotlib qt4`. In the ipython terminal run:

.. code-block:: python

    import hyperspy.api as hs
    import atomap.api as am

If this works, continue with the :ref:`finding_atom_lattices`.
If you get some kind of error, please report it as a New issue on the `Atomap GitLab <https://gitlab.com/atomap/atomap/issues>`_.
Note, having the system and pip version of matplotlib installed at the same might cause an error with matplotlib not finding matplotlib.external.
The easiest way of fixing this is by removing the system version of matplotlib.

.. _install_windows:

Installing in Windows
---------------------

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

If this works, continue with the :ref:`finding_atom_lattices`.
If you get some kind of error, please report it as a New issue on the `Atomap GitLab <https://gitlab.com/atomap/atomap/issues>`_.


Alternative Windows installation
################################

If you already have HyperSpy running in an Anaconda Python environment `Anaconda environment <https://www.continuum.io/downloads>`_,
Atomap can be installed from the `Anaconda prompt` (Start menu - Anaconda3), this will open a command line prompt.
In this prompt run:

.. code-block:: bash

    $ pip install atomap


Development version
-------------------

Grab the development version using the version control system git:

.. code-block:: bash

    $ git clone https://gitlab.com/atomap/atomap.git

Then install it using pip:

.. code-block:: bash

    $ cd atomap
    $ pip3 install -e .
