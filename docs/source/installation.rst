Installation guide
==================

C++ supported compilers:

|gcc version|

|clang version|

|msvc version|

The `plasticity` project is written in `C++` and it supports also older standard versions (std=c++1+).
The package installation can be performed via [`CMake`](https://github.com/Nico-Curti/plasticity/blob/master/CMakeLists.txt).
The `CMake` installer provides also a `plasticity.pc`, useful if you want link to the `plasticity` using `pkg-config`.

You can also use the `plasticity` package in `Python` using the `Cython` wrap provided inside this project.
The only requirements are the following:

* numpy >= 1.16.0
* cython >= 0.29
* scikit -learn >= 0.19.1

The `Cython` version can be built and installed via `CMake` enabling the `-DPYWRAP` variable.
You can use also the `plasticity` package in `Python` using the `Cython` wrap provided inside this project.
The `Python` wrap guarantees also a good integration with the other common Machine Learning tools provided by `scikit-learn` `Python` package; in this way you can use the `plasticity` algorithm as an equivalent alternative also in other pipelines.
Like other Machine Learning algorithm also the `plasticity` one depends on many parameters, i.e its hyper-parameters, which has to be tuned according to the given problem.
The `Python` wrap of the library was written according to `scikit-optimize` `Python` package to allow an easy hyper-parameters optimization using the already implemented classical methods.


.. |gcc version| image:: https://img.shields.io/badge/gcc-4.8.5%20|%204.9.*%20|%205.*%20|%206.*%20|%207.*%20|%208.*%20|%209.*-yellow.svg
.. |clang version| image:: https://img.shields.io/badge/clang-3.*%20|%204.*%20|%205.*%20|%206.*%20|%207.*%20|-red.svg
.. |msvc version| image:: https://img.shields.io/badge/msvc-vs2017%20x86%20|%20vs2017%20x64|%20vs2019%20x86%20|%20vs2019%20x64-blue.svg
.. _CMake: https://github.com/Nico-Curti/plasticity/blob/master/CMakeLists.txt
.. _Makefile: https://github.com/Nico-Curti/plasticity/blob/master/Makefile

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   CMake
   Python
