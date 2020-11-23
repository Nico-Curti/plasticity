| **Authors**  | **Project** |  **Build Status** | **Code Quality** | **Coverage** |
|:------------:|:-----------:|:-----------------:|:----------------:|:------------:|
| [**N. Curti**](https://github.com/Nico-Curti) <br/> [**S. Gasperini**](https://github.com/SimoneGasperini) <br/> [**M. Ceccarelli**](https://github.com/Mat092)  |  **plasticity**  | **Linux/MacOS** : [![travis](https://travis-ci.com/Nico-Curti/plasticity.svg?token=7QqsqaQiuDHSyGDT3xek&branch=main)](https://travis-ci.com/Nico-Curti/plasticity) <br/> **Windows** : [![appveyor](https://ci.appveyor.com/api/projects/status/djnkyxc64dlm4r6p/branch/main?svg=true)](https://ci.appveyor.com/project/Nico-Curti/plasticity-9jr6a/branch/main) | **Codacy** : [![Codacy Badge](https://app.codacy.com/project/badge/Grade/9879f0e8f90140eab79c338b46c00420)](https://www.codacy.com/gh/Nico-Curti/plasticity/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Nico-Curti/plasticity&amp;utm_campaign=Badge_Grade) <br/> **Codebeat** : [![codebeat badge](https://codebeat.co/badges/941ebbcf-de5a-4ff0-b4c4-9674bfb20c69)](https://codebeat.co/projects/github-com-nico-curti-plasticity-main) | [![codecov](https://codecov.io/gh/Nico-Curti/plasticity/branch/master/graph/badge.svg)](https://codecov.io/gh/Nico-Curti/plasticity) |

[![plasticity C++ CI](https://github.com/Nico-Curti/plasticity/workflows/plasticity%20C++%20CI/badge.svg)](https://github.com/Nico-Curti/plasticity/actions?query=workflow%3A%22plasticity+C%2B%2B+CI%22)
[![plasticity Python CI](https://github.com/Nico-Curti/plasticity/workflows/plasticity%20Python%20CI/badge.svg)](https://github.com/Nico-Curti/plasticity/actions?query=workflow%3A%22plasticity+Python+CI%22)
[![plasticity Docs CI](https://github.com/Nico-Curti/plasticity/workflows/plasticity%20Docs%20CI/badge.svg)](https://github.com/Nico-Curti/plasticity/actions?query=workflow%3A%22plasticity+Docs+CI%22)

[![docs](https://readthedocs.org/projects/plasticity/badge/?version=latest)](https://plasticity.readthedocs.io/en/latest/?badge=latest)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/Nico-Curti/plasticity.svg?style=plastic)](https://github.com/Nico-Curti/plasticity/pulls)
[![GitHub issues](https://img.shields.io/github/issues/Nico-Curti/plasticity.svg?style=plastic)](https://github.com/Nico-Curti/plasticity/issues)

[![GitHub stars](https://img.shields.io/github/stars/Nico-Curti/plasticity.svg?label=Stars&style=social)](https://github.com/Nico-Curti/plasticity/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/Nico-Curti/plasticity.svg?label=Watch&style=social)](https://github.com/Nico-Curti/plasticity/watchers)

<a href="https://github.com/UniboDIFABiophysics">
  <div class="image">
    <img src="https://cdn.rawgit.com/physycom/templates/697b327d/logo_unibo.png" width="90" height="90">
  </div>
</a>

# Plasticity

## Unsupervised Neural Networks with biological-inspired learning rules

Implementation and optimization of biological-inspired Neural Network models for the features encoding.

* [Overview](#overview)
* [Theory](#theory)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Efficiency](#efficiency)
* [Usage](#usage)
* [Testing](#testing)
* [Table of contents](#table-of-contents)
* [Contribution](#contribution)
* [References](#references)
* [Authors](#authors)
* [License](#license)
* [Acknowledgments](#acknowledgments)
* [Citation](#citation)

## Overview

Despite the great success of backpropagation algorithm in deep learning, a question remains to what extent the computational properties of artificial neural networks are comparable to the plasticity rules of the human brain.
Indeed, even if the architectures of real and artificial neural networks are similar, the supervised training based on backpropagation and the neurobiology learning rules are unrelated.

In the paper by [D. Krotov and J. J. Hopfield](https://arxiv.org/abs/1806.10181), it is proposed an unusual learning rule, which has a degree of biological plausibility and which is motivated by different well known ideas in neuroplasticity theory:

* Hebb's rule: changes of the synapse strength depend only on the activities of the pre- and post-synaptic neurons and so the learning is **physically local** and describable by local mathematics

* the core of the learning procedure is **unsupervised** because it is believed to be mainly observational, with few or no labels and no explicit task

Starting from these concepts, they were able to design an algorithm (based on an extension of the *Oja rule*) capable of learning early feature detectors in a completely unsupervised way and then use them to train higher-layer weights in a usual supervised neural network.
In particular, the Hopfield model has the structure of a 2-layers neural network which can be described by the following equations:

<img src="https://render.githubusercontent.com/render/math?math=\left\{\begin{matrix}h_j=f(\sum_iw_{ij}v_i)\\c_k=tanh(\sum_js_{jk}h_j)\end{matrix}\right.">

where

<img src="https://render.githubusercontent.com/render/math?math=f(x)=\left\{\begin{matrix}x^n,x\geq0\\0,x<0\end{matrix}\right.">

is the activation function of the unsupervised layer (ReLu for n=1), v<sub>i</sub>, h<sub>j</sub>, c<sub>k</sub> are respectively the input, hidden and output neurons and w<sub>ij</sub>, s<sub>jk</sub> are the receptive fields of the hidden layer (learned by the local unsupervised algorithm) and the weights learned by conventional supervised technique.

## Theory

In this project, a parallel approach founded on the same concepts is proposed.
In particular, it has been developed a model based on the *BCM theory* (E. Bienenstock, L. Cooper, and P. Munro). An exhaustive theoretical description is provided by the original paper of [Castellani et al.](https://pubmed.ncbi.nlm.nih.gov/10378187/).

In general terms, BCM model proposes a sliding threshold for long-term potentiation (LTP) or long-term depression (LTD) induction, and states that synaptic plasticity is stabilized by a dynamic adaptation of the time-averaged postsynaptic activity.
The BCM learning rule is described by the following equations:

<img src="https://render.githubusercontent.com/render/math?math=\frac{dw_i}{dt}=y(y-\theta_M)x_i\sigma'(y)">

<img src="https://render.githubusercontent.com/render/math?math=\theta_M=E[y^2]">

where <img src="https://render.githubusercontent.com/render/math?math=E[\cdot]"> is the time-average operator and, taking the input `x`, the output `y` is computed as:

<img src="https://render.githubusercontent.com/render/math?math=y=\sigma\left(\sum_iw_ix_i\right)">

In this case, the activation function <img src="https://render.githubusercontent.com/render/math?math=\sigma"> is a sigmoidal.

See [here](https://github.com/Nico-Curti/plasticity/blob/master/docs/source/theory.rst) for further details about the models.

## Prerequisites

C++ supported compilers:

![gcc version](https://img.shields.io/badge/gcc-4.8.5%20|%204.9.*%20|%205.*%20|%206.*%20|%207.*%20|%208.*%20|%209.*-yellow.svg)

![clang version](https://img.shields.io/badge/clang-3.6%20|3.9%20|5.*%20|%206.*%20|%207.*%20|-red.svg)

![msvc version](https://img.shields.io/badge/msvc-vs2017%20x86%20|%20vs2017%20x64|%20vs2019%20x86%20|%20vs2019%20x64-blue.svg)

The `plasticity` project is written in `C++` using simple c++14 features.
The package installation can be performed via [`CMake`](https://github.com/Nico-Curti/plasticity/blob/master/CMakeLists.txt).

The `CMake` installer provides also a `plasticity.pc`, useful if you want link to the `plasticity` using `pkg-config`.

You can also use the `plasticity` package in `Python` using the `Cython` wrap provided inside this project.
The only requirements are the following:

* numpy >= 1.15
* cython >= 0.29
* scikit-learn >= 0.20.3
* tqdm
* matplotlib

The `Cython` version can be built and installed via `CMake` enabling the `-DPYWRAP` variable.
The `Python` wrap guarantees also a good integration with the other common Machine Learning tools provided by `scikit-learn` `Python` package; in this way you can use the `plasticity` algorithm as an equivalent alternative also in other pipelines.
Like other Machine Learning algorithm also the `plasticity` one depends on many parameters, i.e its hyper-parameters, which has to be tuned according to the given problem.
The `Python` wrap of the library was written according to `scikit-optimize` `Python` package to allow an easy hyper-parameters optimization using the already implemented classical methods.

## Installation

Follow the instruction about your needs.

A complete list of instructions "for beginners" is also provided for both [cpp](https://github.com/Nico-Curti/plasticity/blob/master/docs/source/CMake.rst) and [python](https://github.com/Nico-Curti/plasticity/blob/master/docs/source/Python.rst) versions.

### CMake C++ installation

We recommend to use `CMake` for the installation since it is the most automated way to reach your needs.
First of all make sure you have a sufficient version of `CMake` installed (3.9 minimum version required).
If you are working on a machine without root privileges and you need to upgrade your `CMake` version a valid solution to overcome your problems is provided [here](https://github.com/Nico-Curti/Shut).

With a valid `CMake` version installed first of all clone the project as:

```bash
git clone https://github.com/Nico-Curti/plasticity
cd plasticity
```

The you can build the `plasticity` package with

```bash
mkdir -p build
cd build && cmake .. && cmake --build . --target install
```

or more easily

```bash
./build.sh
```

if you are working on a Windows machine the right script to call is the [`build.ps1`](https://Nico-Curti/plasticity/blob/master/build.ps1).

**NOTE 1:** the only requirement of the library is Eigen3. Please pay attention to install this dependency before running the CMake installation to avoid any issue.

**NOTE 2:** if you want enable the OpenMP support (*4.5 version is required*) compile the library with `-DOMP=ON`.

### Python installation

Python version supported : ![Python version](https://img.shields.io/badge/python-3.5|3.6|3.7|3.8-blue.svg)

The easiest way to install the package is to use `pip`

```bash
python -m pip install plasticity
```

> :warning: The setup file requires the `Cython` and `Numpy` packages, thus make sure to pre-install them!
> We are working on some workarounds to solve this issue.

The `Python` installation can be performed with or without the `C++` installation.
The `Python` installation is always executed using [`setup.py`](https://github.com/Nico-Curti/plasticity/blob/master/setup.py) script.

If you have already built the `plasticity` `C++` library the installation is performed faster and the `Cython` wrap was already built using the `-DPYWRAP` definition.
Otherwise the full list of dependencies is build.

In both cases the installation steps are

```bash
python -m pip install -r ./requirements.txt
```

to install the prerequisites and then

```bash
python setup.py install
```

or for installing in development mode:

```bash
python setup.py develop --user
```

> :warning: The current installation via pip has no requirements about the version of `setuptools` package.
> If the already installed version of `setuptools` is `>= 50.*` you can find some troubles during the installation of our package (ref. [issue](https://github.com/Nico-Curti/rFBP/issues/5)).
> We suggest to temporary downgrade the `setuptools` version to `49.3.0` to workaround this `setuptools` issue.

## Efficiency

![Comparison of time performances between the pure-`Python` implementation and `Cython` version of the `BCM` model varying the input dimension sizes (number of samples and number of features). For each input configuration 10 runs of both algorithm were performed keeping fixed all the other parameters. In the simulation we used 10 epochs and a SGD optimization algorithm.](./img/BCM_timing.png)

![Comparison of time performances between the pure-`Python` implementation and `Cython` version of the `Hopfield` model varying the input dimension sizes (number of samples and number of features). For each input configuration 10 runs of both algorithm were performed keeping fixed all the other parameters. In the simulation we used 10 epochs and a SGD optimization algorithm.](./img/Hopfield_timing.png)

We test the computational efficiency of both the implementation (pure-`Python` and `Cython` with multi-threading enabled).
The tests were performed keeping fixed all the training parameters and varying just the input dimension (number of samples and number of features).

As expected we have the most significant improvements enlarging the number of samples, while varying the number of features the scalability of the code is quite stable.
Both the algorithms spend the same time for the simulations and therefore their use is not constrained by any computational limitation.
We however encourage the use of the `Cython` version since it is obviously faster than the `Python` counterpart.

## Usage

You can use the `plasticity` library into pure-Python modules or inside your C++ application.

### C++ Version

**TODO**

### Python Version

The `plasticity` classes are totally equivalent to a `scikit-learn` feature-encoder method and thus they provide the member functions `fit` (to train your model) and `predict` (to test a trained model on new samples).
First of all you need to import the desired `plasticity` class and then simply call the training/testing functions.

```python
from plasticity.models import BCM
from sklearn.datasets import fetch_openml

# Download the MNIST dataset
X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)

# normalize the sample into [0, 1]
X *= 1. / 255

from plasticity.model import BCM

model = BCM(outputs=100, num_epochs=10, batch_size=100, interaction_strenght=0.)
model.fit(X)
```

You can also visualize the weights connections using the utility functions provided by the package as

```python

from plasticity.utils import view_weights

view_weights (model.weights, dims=(28, 28))
```

The results should be something like

<a href="https://github.com/Nico-Curti/plasticity/blob/master/img/BCM_weights.gif">
  <div class="image">
    <img src="img/BCM_weights.gif" width="392" height="392">
  </div>
</a>


## Testing

**TODO**

## Table of contents

Description of the folders related to the `C++` version.

| **Directory**  |  **Description** |
|:--------------:|:-----------------|
| [hpp](https://github.com/Nico-Curti/plasticity/blob/master/hpp)         | Implementation of the C++ template functions and objects used in the `plasticity` library |
| [include](https://github.com/Nico-Curti/plasticity/blob/master/include) | Definition of the C++ function and objects used in the `plasticity` library |
| [src](https://github.com/Nico-Curti/plasticity/blob/master/src)         | Implementation of the C++ functions and objects used in the `plasticity` library |

Description of the folders related to the `Python` version.

| **Directory**  |  **Description** |
|:--------------:|:-----------------|
| [example](https://github.com/Nico-Curti/plasticity/blob/master/plasticity/example) | `Jupyter` notebook with some examples on the MNIST (digit) dataset. |
| [lib](https://github.com/Nico-Curti/plasticity/blob/master/plasticity/lib)         | List of `Cython` definition files |
| [source](https://github.com/Nico-Curti/plasticity/blob/master/plasticity/source)   | List of `Cython` implementation objects |
| [model](https://github.com/Nico-Curti/plasticity/blob/master/plasticity/model)     | pure-`Python` implementation of the classes |
| [cython](https://github.com/Nico-Curti/plasticity/blob/master/plasticity/cython)   | `Cython`-wraps of the classes |

## Contribution

Any contribution is more than welcome :heart:. Just fill an [issue](https://github.com/Nico-Curti/plasticity/blob/master/.github/ISSUE_TEMPLATE/ISSUE_TEMPLATE.md) or a [pull request](https://github.com/Nico-Curti/plasticity/blob/master/.github/PULL_REQUEST_TEMPLATE/PULL_REQUEST_TEMPLATE.md) and we will check ASAP!

See [here](https://github.com/Nico-Curti/plasticity/blob/master/.github/CONTRIBUTING.md) for further informations about how to contribute with this project.

## References

<blockquote>1- Castellani G., Intrator N., Shouval H.Z., Cooper L.N. Solutions of the BCM learning rule in a network of lateral interacting nonlinear neurons, Network Computation in Neural Systems, 10.1088/0954-898X/10/2/001. </blockquote>

<blockquote>2- Dmitry Krotov, and John J. Hopfield. Unsupervised learning by competing hidden units, PNAS, 2019, www.pnas.org/cgi/doi/10.1073/pnas.1820458116. </blockquote>

## Authors

* <img src="https://avatars0.githubusercontent.com/u/24650975?s=400&v=4" width="25px"> **Nico Curti** [git](https://github.com/Nico-Curti), [unibo](https://www.unibo.it/sitoweb/nico.curti2)

* <img src="https://avatars2.githubusercontent.com/u/71086758?s=400&v=4" width="25px;"/> **Simone Gasperini** [git](https://github.com/SimoneGasperini)

* <img src="https://avatars0.githubusercontent.com/u/41483077?s=400&v=4" width="25px;"/> **Mattia Ceccarelli** [git](https://github.com/Mat092)

See also the list of [contributors](https://github.com/Nico-Curti/plasticity/contributors) [![GitHub contributors](https://img.shields.io/github/contributors/Nico-Curti/plasticity.svg?style=plastic)](https://github.com/Nico-Curti/plasticity/graphs/contributors/) who participated in this project.

## License

The `plasticity` package is licensed under the MIT "Expat" License. [![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/Nico-Curti/plasticity/blob/master/LICENSE)

## Acknowledgments

Thanks goes to all contributors of this project.

We thank also the author(s) of [Catch2](https://github.com/catchorg/Catch2) library: we have used it in the testing procedure of our C++ version and it is amazing!

## Citation

If you have found `plasticity` helpful in your research, please consider citing the project repository

```BibTeX
@misc{plasticity,
  author = {Curti, Nico and Gasperini, Simone and Ceccarelli, Mattia},
  title = {plasticity - Unsupervised Neural Networks with biological-inspired learning rules},
  year = {2020},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Nico-Curti/plasticity}},
}
```
