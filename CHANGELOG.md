# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2020-11-05

First version of the algorithm.
The starting point of this project is given by the https://github.com/SimoneGasperini/biological_neuralnet repository.

### Added

- First C++ version of the BCM algorithm
- Add cython wrap for the Python support
- Add scikit-learn compatibility for the classes in the model sub-package
- Add sphinx + doxygen support for the documentation
- Add an utility sub-package for common functions
- Add setup installation of the package (serial)
- Add first version of CI using travis and appveyor
- Add first version of code evaluation with codebeat and codacy
- Add optimizer object for the convergency of the training

### Changed

- Implement the class inheritance in the BCM and Hopfield models
- Use a set of Activation classes for improve the model testing

### Improvements

- Use the numpy einsum function for an optimized implementation of the GEMM
- Use the Eigen3 library for the matrix inversion in the C++ implementation of the BCM
- Use the OpenMP support for a parallel computation of the training

### TODO

- Improve the compatibility of the setup script for multiple OS (maybe using scikit-build)
- Add the interaction_matrix GEMM in the C++ version of the BCM
- Implement the setup run into the CMake installation
- Implement a series of tests for the package CI (see coverage)
- Upload the package to PyPi at the first release
- Fix sphinx documentation
