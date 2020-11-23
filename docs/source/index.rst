.. plasticity algorithm documentation master file, created by
   sphinx-quickstart on Fri Oct  2 12:42:24 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to plasticity library's documentation!
==================================================================


Overview
========


Usage example
=============

You can use the `plasticity` library into pure-Python modules or inside your C++ application.

C++ example
-----------


Python example
--------------

The `plasticity` classes are totally equivalent to a `scikit-learn` feature-encoder method and thus they provide the member functions `fit` (to train your model) and `predict` (to test a trained model on new samples).

First of all you need to import the desired `plasticity` class and then simply call the training/testing functions.

.. code-block:: python

  from plasticity.models import BCM
  from sklearn.datasets import fetch_openml

  # Download the MNIST dataset
  X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)

  # normalize the sample into [0, 1]
  X *= 1. / 255

  from plasticity.model import BCM

  model = BCM(outputs=100, num_epochs=10, batch_size=100, interaction_strenght=0.)
  model.fit(X)

You can also visualize the weights connections using the utility functions provided by the package as

.. code-block:: python

  from plasticity.utils import view_weights

  view_weights (model.weights, dims=(28, 28))

The results should be something like

.. image:: ../../img/BCM_weights.gif

For sake of completeness the same pipeline can be run also with the Hopfield model.
In this case the could will be

.. code-block:: python

  from plasticity.models import Hopfield
  from sklearn.datasets import fetch_openml

  # Download the MNIST dataset
  X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)

  # normalize the sample into [0, 1]
  X *= 1. / 255

  from plasticity.model import Hopfield

  model = Hopfield(outputs=100, num_epochs=10, batch_size=100)
  model.fit(X)

And in this case the weights matrix should appear similar to

.. image:: ../../img/Hopfield_weights.gif


.. _`scikit-learn`: https://github.com/scikit-learn/scikit-learn
.. _`scikit-optimize`: https://github.com/scikit-optimize/scikit-optimize


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   theory
   installation
   cppAPI/modules
   pyAPI/modules
   examples/modules
   references
