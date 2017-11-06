.. _development_guide:

This is a technical guide for people already familiar with scientific Python software development.
For an introduction to scientific Python software development, see :ref:`contribute`.

=================
Development guide
=================

Testing
-------

Atomap has a large number of unit tests, which is tested using ``pytest``:

.. code-block:: bash

    $ python3 -m pytest --doctest-modules atomap/

This also runs every docstring example as a unit test.

The documentation is tested by doing:

.. code-block:: bash

    $ python3 -m pytest --doctest-glob="*.rst" doc/


Both the unit tests and the doc tests can be sped up by running the tests parallel.
Use the `xdist <https://docs.pytest.org/en/3.0.0/xdist.html>`_ pytest package for this.
To run the tests using 5 parallel processes:

.. code-block:: bash

    $ python3 -m pytest -n 5 --doctest-modules atomap/
    $ python3 -m pytest -n 5 --doctest-glob="*.rst" doc/


Style checks
------------

.. code-block:: bash

    $ python3 -m flake8 --exclude atomap/api.py atomap/


Generating the sphinx page
--------------------------

.. code-block:: bash

    $ cd doc
    $ python3 -m sphinx -b html ./ _build/html/


Continuous integration
----------------------

The Continuous integration (CI) settings is contained in ``.gitlab-ci.yml``.
This runs all the above-mentioned tests, style checks and sphinx page generation on each branch.


Documentation from development branch
-------------------------------------

The most recent documentation generated from the development branch can be accessed `here <https://gitlab.com/atomap/atomap/builds/artifacts/master/file/pages_development/index.html?job=pages_development_branch>`_.
