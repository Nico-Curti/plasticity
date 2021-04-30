# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2021-04-30

Porting of the full algorithm to the Eigen library.

### Added

- (C++) MNIST dataset loader class
- (C++) Configuration file parser for the simulation
- (C++) Add the OpenCV support for the weights visualization
- (C++) Add (custom) `bwr` OpenCV colormap
- (C++) Add examples for BCM and Hopfield usage with the MNIST dataset
- (C++) Add version check utility in the examples
- (C++) Add fit callback support for the visualization of the learning weights
- (C++) Add a brief list of test for the BCM and Hopfield models

### Changed

- (C++) **Move** the activation function namespace to `transfer_t`
- (C++) **Move** the optimizer function namespace to `optimizer_t`
- (C++) **Move** the weights initialization function namespace to `weights_init_t`
- (C++) **Remove** OpenMP support
- (Global) Split the github-actions for the C++ support into different configuration files

### Improvements

- (C++) Revision of the BCM algorithm with the Eigen support
- (C++) Revision of the Hopfield algorithm with the Eigen support
- (C++) Revision of the Optimization algorithms with the Eigen support
- (Global) Improve the README documentation
- (Global) Minor fix on the Doxygen documentation
- (Global) Add FindNumPy CMake module from scikit-build (ref. [FindNumPy.cmake](https://github.com/scikit-build/scikit-build/blob/master/skbuild/resources/cmake/FindNumPy.cmake))

### TODO

- (C++) Check the convergency method of the models
- (C++) Improve the list of testing functions
- (Python) Implement a series of tests for the package CI (see coverage)
- (Global) Upload the package to PyPi at the first release
- (Global) Improve/Update the documentation of the project on Read-the-Docs
- (Global) Improve README documentation about the models' theories
- (Global) Fix Doxygen error on lambda function as default argument

--------------------------------------------------------------------------------------------------


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
