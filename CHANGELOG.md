# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2020-11-17

First version of the algorithm.

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
- Add inpainting examples using BCM and Hopfield models
- Add classifier examples using BCM and Hopfield models
- Testing the performances of both the models on the MNIST dataset (without supervised part!!)
- Add multiple initializers for the weights matrix since we notice different convergency behaviour changing the initial conditions

### Changed

- Implement the class inheritance in the BCM and Hopfield models
- Use a set of Activation classes for improve the model testing
- The stop criteria is based on the theta array for the BCM model and on the xx array in the Hopfield model
- The stop criteria monitors the absolute difference despite the relative difference

### Improvements

- Use the numpy einsum function for an optimized implementation of the GEMM
- Use the Eigen3 library for the matrix inversion in the C++ implementation of the BCM
- Use the OpenMP support for a parallel computation of the training
- Add stopping criteria in both the C++ and Python versions

### TODO

- Implement a series of tests for the package CI (see coverage)
- Upload the package to PyPi at the first release
