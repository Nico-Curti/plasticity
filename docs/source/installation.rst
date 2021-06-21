Installation guide
==================

C++ supported compilers:

|gcc version|

|clang version|

|msvc version|

The `plasticity` project is written in `C++` and it supports also older standard versions (std=c++1+).
The package installation can be performed via CMake_.
The only requirement for the installation of the `C++` library is the `Eigen3` library.
You can easily install the `Eigen3` library with the following commands:

+--------------+-----------------------------------+
| **OS**       |  **Command**                      |
+==============+===================================+
| **Linux**    | `sudo apt install libeigen3-dev`  |
+--------------+-----------------------------------+
| **MacOS**    | `brew install eigen`              |
+--------------+-----------------------------------+
| **Windows**  | `vcpkg install eigen3`            |
+--------------+-----------------------------------+

You can also use the `plasticity` package in `Python` using the `Cython` wrap provided inside this project.
The only requirements are the following:

* numpy >= 1.16.0
* cython >= 0.29
* scikit -learn >= 0.19.1
* tqdm
* matplotlib

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

Follow the instruction about your needs.

.. |gcc version| image:: https://img.shields.io/badge/gcc-4.8.5%20|%204.9.*%20|%205.*%20|%206.*%20|%207.*%20|%208.*%20|%209.*%20|10.*-yellow.svg
.. |clang version| image:: https://img.shields.io/badge/clang-3.8+%20|%204.*%20|%205.*%20|%206.*%20|%207.*%20|%208.*%20|%209.*%20|%2010.*-red.svg
.. |msvc version| image:: https://img.shields.io/badge/msvc-vs2017%20x86%20|%20vs2017%20x64|%20vs2019%20x86%20|%20vs2019%20x64-blue.svg
.. _CMake: https://github.com/Nico-Curti/plasticity/blob/main/CMakeLists.txt

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   CMake
   Python
