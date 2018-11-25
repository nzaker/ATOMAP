.. _install:

==========
Installing
==========

.. _install_windows:

Installing in Windows
---------------------

Anaconda Python environment
***************************

Currently, the easiest way to install Atomap is using the Anaconda python environment `Anaconda environment <https://www.continuum.io/downloads>`_,
Install HyperSpy, then Atomap via the `Anaconda prompt` (Start menu - Anaconda3), this will open a command line prompt.
In this prompt run:

.. code-block:: bash

    $ conda install hyperspy -c conda-forge
    $ pip install atomap

If everything installed, continue to :ref:`start_atomap_windows`.
If you got some kind of error, please report it as a New issue on the `Atomap GitLab <https://gitlab.com/atomap/atomap/issues>`_.


WinPython HyperSpy installer
****************************

Alternatively, the WinPython HyperSpy bundle can be used.
Firstly download and install the `WinPython HyperSpy bundle <https://github.com/hyperspy/hyperspy-bundle/releases>`_:

After installing the bundle, there should be a folder in the start menu called "HyperSpy Bundle", and this
folder should contain the "WinPython prompt". Start the "WinPython prompt". This will open a terminal window called
"WinPython prompt", in this window type and run:

.. code-block:: bash

    pip install atomap

If everything installed, continue to :ref:`start_atomap_windows`.
If you got some kind of error, please report it as a New issue on the `Atomap GitLab <https://gitlab.com/atomap/atomap/issues>`_.


Installing in MacOS
-------------------

Install the Anaconda python environment: `Anaconda environment <https://www.continuum.io/downloads>`_, and through the `Anaconda prompt` install HyperSpy and Atomap:

.. code-block:: bash

    $ conda install hyperspy -c conda-forge
    $ pip install atomap


If everything installed, continue to :ref:`start_atomap_macos`.
If you got some kind of error, please report it as a New issue on the `Atomap GitLab <https://gitlab.com/atomap/atomap/issues>`_.


Installing in Linux
-------------------

The recommended way to install Atomap is using PIP, which is a package manager for python.
It is recommended to first install the precompiled dependencies using the system package manager.

`HyperSpy <http://hyperspy.org/>`_ is also included as Atomap relies heavily on the modelling and visualization functionality in HyperSpy.

Ubuntu 18.04
************

.. code-block:: bash

    $ sudo apt-get install ipython3 python3-pip python3-numpy python3-scipy python3-matplotlib python3-sklearn python3-skimage python3-h5py python3-dask python3-traits python3-tqdm python3-pint python3-dask python3-pyqt5 python3-lxml python3-sympy python3-sparse python3-statsmodels python3-numexpr python3-ipykernel python3-jupyter-client python3-requests python3-dill python3-natsort
    $ pip3 install --user atomap

If everything installed, continue to :ref:`start_atomap_linux`.
If you got some kind of error, please report it as a New issue on the `Atomap GitLab <https://gitlab.com/atomap/atomap/issues>`_.


Development version
-------------------

Grab the development version using the version control system git (see :ref:`contribute`):

.. code-block:: bash

    $ git clone https://gitlab.com/atomap/atomap.git

Then install it using pip:

.. code-block:: bash

    $ cd atomap
    $ pip3 install -e .
