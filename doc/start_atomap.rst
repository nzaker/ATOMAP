.. _start_atomap:


============
Start Atomap
============

Starting Python
---------------

The first step is starting an interactive Jupyter Notebook environment.

.. _start_atomap_linux:

Linux
^^^^^

Open a terminal and start ``ipython3``:

.. code-block:: bash

    $ ipython3 notebook


If ``ipython3`` is not available, try ``ipython``:

.. code-block:: bash

    $ ipython notebook


This will open a browser window (or a new browser tab).
Press the "New" button (top right), and start a Python 3 Notebook.
In the first cell, run the following commands (paste them, and press Shift + Enter).
If you are unfamiliar with the Jupyter Notebook interface, `see the Jupyter Notebook guide <https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb>`_.

.. code-block:: python

    %matplotlib nbagg
    import atomap.api as am

If this works, continue to the :ref:`tutorials`.
If you get some kind of error, please report it as a New issue on the `Atomap GitLab <https://gitlab.com/atomap/atomap/issues>`_.


.. _start_atomap_windows:

Windows
^^^^^^^

This depends on the installation method:

* If the HyperSpy bundle was installed, go to the "HyperSpy Bundle" in the start-menu and start "Jupyter Notebook".
* If Anaconda was used, go to "Anaconda3" in the start menu, and start "Jupyter Notebook".

This will open a browser window (or a new browser tab).
Press the "New" button (top right), and start a Python 3 Notebook.
In the first cell, run the following commands (paste them, and press Shift + Enter).
If you are unfamiliar with the Jupyter Notebook interface, `see the Jupyter Notebook guide <https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb>`_.

.. code-block:: python

    %matplotlib nbagg
    import atomap.api as am

If this works, continue to the :ref:`tutorials`.
If you get some kind of error, please report it as a New issue on the `Atomap GitLab <https://gitlab.com/atomap/atomap/issues>`_.


.. _start_atomap_macos:

MacOS
^^^^^

Open the Terminal, and write:

.. code-block:: bash

    $ jupyter notebook


This will open a browser window (or a new browser tab).
Press the "New" button (top right), and start a Python 3 Notebook.
In the first cell, run the following commands (paste them, and press Shift + Enter).
If you are unfamiliar with the Jupyter Notebook interface, `see the Jupyter Notebook guide <https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb>`_.

.. code-block:: python

    %matplotlib nbagg
    import atomap.api as am

If this works, continue to the :ref:`tutorials`.
If you get some kind of error, please report it as a New issue on the `Atomap GitLab <https://gitlab.com/atomap/atomap/issues>`_.


.. _tutorials:

Tutorials
---------

To get you started on using Atomap there are tutorials available.
The first tutorial :ref:`finding_atom_lattices` aims at showing how atom positions are found, while :ref:`analysing_atom_lattices` shows how this information can be visualized.
There is also a tutorial showing how you can make your analysis semi-automatic, :ref:`automation`.

The `>>>` used in the tutorials and documentation means the comment should be typed inside some kind of Python prompt, and can be copy-pasted directly into the *Jupyter Notebooks*.


Atomap demos
^^^^^^^^^^^^

In addition to the guides on this webpage, another good resource is the `Atomap demos <https://gitlab.com/atomap/atomap_demos/>`_, which are pre-filled Jupyter Notebooks showing various aspects of Atomap's functionality.
For beginners, the `Introduction to Atomap notebook <https://gitlab.com/atomap/atomap_demos/blob/release/introduction_to_atomap.ipynb>`_ is a good place to start.
