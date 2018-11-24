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


To check everything is working correctly, go to "Anaconda3" in the start menu, and start "Jupyter Notebook".
This will open a browser window (or a new browser tab).
Start a new Python 3 notebook, and run in the first cell:

.. code-block:: python

    %matplotlib nbagg
    import atomap.api as am


If this works, continue with the :ref:`finding_atom_lattices`.
If you get some kind of error, please report it as a New issue on the `Atomap GitLab <https://gitlab.com/atomap/atomap/issues>`_.


WinPython HyperSpy installer
****************************

Alternatively, the WinPython HyperSpy bundle can be used.
Firstly download and install the `WinPython HyperSpy bundle <https://github.com/hyperspy/hyperspy-bundle/releases>`_:

After installing the bundle, there should be a folder in the start menu called "HyperSpy WinPython Bundle", and this
folder should contain the "WinPython prompt". Start the "WinPython prompt". This will open a terminal window called
"WinPython prompt", in this window type and run:

.. code-block:: bash

    pip install atomap

To check everything is working correctly, go to the "HyperSpy WinPython Bundle" and start "Jupyter QtConsole".
This will open a new window. In this window, run:

.. code-block:: python

    %matplotlib nbagg
    import hyperspy.api as hs
    import atomap.api as am

If this works, continue with the :ref:`finding_atom_lattices`.
If you get some kind of error, please report it as a New issue on the `Atomap GitLab <https://gitlab.com/atomap/atomap/issues>`_.


Installing in Linux
-------------------

The recommended way to install Atomap is using PIP, which is a package manager for python.
It is recommended to first install the precompiled dependencies using the system package manager.

`HyperSpy <http://hyperspy.org/>`_ is also included as Atomap relies heavily on the modelling and visualization functionality in HyperSpy.

Ubuntu 17.10
************

.. code-block:: bash

    $ sudo apt-get install ipython3 python3-pip python3-numpy python3-scipy python3-matplotlib python3-sklearn python3-skimage python3-h5py python3-dask python3-traits python3-tqdm python3-pint python3-dask python3-pyqt5 python3-lxml
    $ sudo apt-get install python3-sympy --no-install-recommends
    $ pip3 install --upgrade pip
    $ pip3 install --user atomap


Starting Atomap
***************

To check that everything is working, open a terminal and run :code:`ipython3 --matplotlib qt5`. In the ipython terminal run:

.. code-block:: python

    import hyperspy.api as hs
    import atomap.api as am

If this works, continue with the :ref:`finding_atom_lattices`.
If you get some kind of error, please report it as a New issue on the `Atomap GitLab <https://gitlab.com/atomap/atomap/issues>`_.
Note, having the system and pip version of matplotlib installed at the same might cause an error with matplotlib not finding matplotlib.external.
The easiest way of fixing this is by removing the system version of matplotlib.


Development version
-------------------

Grab the development version using the version control system git:

.. code-block:: bash

    $ git clone https://gitlab.com/atomap/atomap.git

Then install it using pip:

.. code-block:: bash

    $ cd atomap
    $ pip3 install -e .
