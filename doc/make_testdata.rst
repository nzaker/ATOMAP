.. _make_testdata:

===========================
Making datasets for testing
===========================

The functionality for generating specific test data is useful for

1. Testing processing tools on known standards
2. Finding parameter sensitivity for processing tools
3. Easily generate example datasets
4. Use in unit tests for Atomap

To generate an image of a simple cubic structure

.. code-block:: python

    >>> import atomap.api as am
    >>> signal = am.get_simple_cubic_signal()

.. image:: images/testdata/testdata_simple_cubic.png
    :scale: 50 %
    :align: center

The sublattice of the simple cubic structure can also be generated

.. code-block:: python

    >>> sublattice = am.get_simple_cubic_sublattice()
    <Sublattice,  (atoms:225,planes:0)>
    
More advanced datasets can also be created.
The functionality for generating testdata signals uses the function sublattice.get_model_image.
