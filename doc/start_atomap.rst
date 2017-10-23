.. _start_atomap:


============
Start Atomap
============

Starting Python
---------------

The first step is starting an interactive Python environment (IPython).

Linux
^^^^^

Open a terminal and start `ipython3`:

.. code-block:: bash

    $ ipython3 --matplotlib qt4

If `ipython3` is not available, try `ipython`:

.. code-block:: bash

    $ ipython --matplotlib qt4

Windows
^^^^^^^

This depends on the installation method.
If Anaconda was used, there should be an *Anaconda3* folder in the start menu.
Start the interactive Python environment, it should be called either *IPython* or *Jupyter QtConsole*.
This will open a command line prompt.
This prompt will be referred to as the *IPython terminal*.

Tutorials
---------

To get you started with using Atomap, and getting and overview and understanding of how Atomap works there are tutorials available.
The first tutorial :ref:`finding_atom_lattices` aims at showing how atom positions are found, while :ref:`analysing_atom_lattices` shows how this information can be visualized.
There is also a tutorial showing how you can make your analysis semi-automatic, :ref:`automation`.

The `>>>` used in the tutorials and documentation means the comment should be typed inside some kind of Python prompt, so do not include these when actually running the code.

Jupyter Notebook
----------------

In addition to the tutorials on this webpage, interactive tutorials in the form of a Jupyter Notebook are available: https://gitlab.com/atomap/atomap_demos/blob/master/notebook_example/Atomap.ipynb
