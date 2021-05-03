Python Installation
===================

Python version supported : |Python version|

The easiest way to install the package is using `pip`

.. code-block:: bash

  python -m pip install plasticity

.. warning::

  The setup file requires the `Cython` package, thus make sure to pre-install it!
  We are working on some workarounds to solve this issue.

The `Python` installation can be performed with or without the `C++` installation.
The `Python` installation is always executed using `setup.py`_ script.

If you have already built the `plasticity` `C++` library the installation is performed faster and the `Cython` wrap was already built using the `-DPYWRAP` definition.
Otherwise the full list of dependencies is build.

In both cases the installation steps are

.. code-block:: bash

  python -m pip install -r ./requirements.txt

to install the prerequisites and then

.. code-block:: bash

  python setup.py install

or for installing in development mode:

.. code-block:: bash

  python setup.py develop --user

.. warning::

  The current installation via pip has no requirements about the version of `setuptools` package.
  If the already installed version of `setuptools` is `>= 50.*` you can find some troubles during the installation of our package (ref. issue_).
  We suggest to temporary downgrade the `setuptools` version to `49.3.0` to workaround this `setuptools` issue.


.. |Python version| image:: https://img.shields.io/badge/python-3.5|3.6|3.7|3.8-blue.svg
.. _`setup.py`: https://github.com/Nico-Curti/plasticity/blob/main/setup.py
.. _issue: https://github.com/Nico-Curti/rFBP/issues/5