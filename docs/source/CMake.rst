CMake C++ Installation
======================

We recommend to use `CMake` for the installation since it is the most automated way to reach your needs.
First of all make sure you have a sufficient version of `CMake` installed (3.9 minimum version required).
The only external dependency of the `C++` code is given by the `Eigen3` library, therefore make sure to have already installed the library (however the `CMake` internally checks the presence of it).
If you are working on a machine without root privileges and you need to upgrade your `CMake` version a valid solution to overcome your problems is provided shut_.

For the C++ installation:

1) Follow your system prerequisites (below)

2) Clone the `plasticity` package from this repository, or download a stable release

.. code-block:: bash

  git clone https://github.com/Nico-Curti/plasticity.git
  cd plasticity

3) `plasticity` could be built with CMake and Make or with the *build* scripts in the project.
Example:

**Unix OS:**

.. code-block:: bash

  ./build.sh Release

**Windows OS:**

.. code-block:: bash

  PS \>                 ./build.ps1 Release

Ubuntu
------

1) Define a work folder, which we will call WORKSPACE in this tutorial: this could be a "Code" folder in our home, a "c++" folder on our desktop, whatever you want. Create it if you don't already have, using your favourite method (mkdir in bash, or from the graphical interface of your distribution). We will now define an environment variable to tell the system where our folder is. Please note down the full path of this folder, which will look like `/home/$(whoami)/code/`

.. code-block:: bash

  echo -e "\n export WORKSPACE=/full/path/to/my/folder \n" >> ~/.bashrc
  source ~/.bashrc

2) Open a Bash terminal and type the following commands to install all the prerequisites.

.. code-block:: bash

  sudo add-apt-repository ppa:ubuntu-toolchain-r/test
  sudo apt-get update
  sudo apt-get install -y gcc-8 g++-8

  wget --no-check-certificate https://cmake.org/files/v3.13/cmake-3.13.1-Linux-x86_64.tar.gz
  tar -xzf cmake-3.13.1-Linux-x86_64.tar.gz
  export PATH=$PWD/cmake-3.13.1-Linux-x86_64/bin:$PATH

  sudo apt-get install -y make git dos2unix ninja-build
  git config --global core.autocrlf input
  git clone https://github.com/physycom/sysconfig

3) Install the Eigen library

.. code-block:: bash

  sudo apt-get install -y libeigen3-dev

4) Build the project with CMake (enable or disable OMP with the define **-DOMP**; enable or disable **Cython** building with the define **-DPYWRAP**; enable or disable testing with the define **-DBUILD_TEST**):

.. code-block:: bash

  cd $WORKSPACE
  git clone https://github.com/Nico-Curti/plasticity
  cd plasticity

  mkdir -p build
  cd build

  cmake -DOMP:BOOL=ON -DPYWRAP:BOOL=ON ..
  make -j
  cmake --build . --target install
  cd ..

macOS
-----

1) If not already installed, install the XCode Command Line Tools, typing this command in a terminal:

.. code-block:: bash

  xcode-select --install

2) If not already installed, install Homebrew following the official guide here: https://brew.sh/index_it.html.

3) Open the terminal and type these commands

.. code-block:: bash

  brew update
  brew upgrade
  brew install gcc@8
  brew install cmake make git ninja

4) Install the Eigen library

.. code-block:: bash

  brew install eigen

5) Define a work folder, which we will call WORKSPACE in this tutorial: this could be a "Code" folder in our home, a "c++" folder on our desktop, whatever you want. Create it if you don't already have, using your favourite method (mkdir in bash, or from the graphical interface in Finder). We will now define an environment variable to tell the system where our folder is. Please note down the full path of this folder, which will look like /home/$(whoami)/code/

6) Open a Terminal and type the following command (replace /full/path/to/my/folder with the previous path noted down)

.. code-block:: bash

  echo -e "\n export WORKSPACE=/full/path/to/my/folder \n" >> ~/.bash_profile
  source ~/.bash_profile

7) Build the project with CMake (enable or disable OMP with the define **-DOMP**; enable or disable **Cython** building with the define **-DPYWRAP**; enable or disable testing with the define **-DBUILD_TEST**):

.. code-block:: bash

  cd $WORKSPACE
  git clone https://github.com/Nico-Curti/plasticity
  cd plasticity

  mkdir -p build
  cd build

  cmake -DOMP:BOOL=ON -DPYWRAP:BOOL=ON ..
  make -j
  cmake --build . --target install
  cd ..

Windows (7+)
------------

1) Install Visual Studio 2017 from the official website here: https://www.visualstudio.com/

2) Open your Powershell with Administrator privileges, type the following command and confirm it:

.. code-block:: bash

  PS \>                 Set-ExecutionPolicy unrestricted

3) If not already installed, please install chocolatey using the official guide here: http://chocolatey.org

4) If you are not sure about having them updated, or even installed, please install `git`, `cmake` and an updated `Powershell`. To do so, open your Powershell with Administrator privileges and type

.. code-block:: bash

  PS \>                 cinst -y git cmake powershell

5) Restart the PC if required by chocolatey after the latest step

6) Install PGI 18.10 from the official website (https://www.pgroup.com/products/community.htm) (the community edition is enough and is free; NOTE: install included MS-MPI, but avoid JRE and Cygwin)

7) Activate license for PGI 18.10 Community Edition (rename the file `%PROGRAMFILES%\PGI\license.dat-COMMUNITY-18.10` to `%PROGRAMFILES%\PGI\license.dat`) if necessary, otherwise enable a Professional License if available

8) Define a work folder, which we will call `WORKSPACE` in this tutorial: this could be a "Code" folder in our home, a "cpp" folder on our desktop, whatever you want. Create it if you don't already have, using your favourite method (mkdir in Powershell, or from the graphical interface in explorer). We will now define an environment variable to tell the system where our folder is. Please note down its full path. Open a Powershell (as a standard user) and type

.. code-block:: bash

  PS \>                 rundll32 sysdm.cpl,EditEnvironmentVariables

9) In the upper part of the window that pops-up, we have to create a new environment variable, with name `WORKSPACE` and value the full path noted down before.
If it not already in the `PATH` (this is possible only if you did it before), we also need to modify the "Path" variable adding the following string (on Windows 10 you need to add a new line to insert it, on Windows 7/8 it is necessary to append it using a `;` as a separator between other records):

.. code-block:: bash

                      %PROGRAMFILES%\CMake\bin

10) If `vcpkg` is not installed, please follow the next procedure, otherwise please jump to #12

.. code-block:: bash

  PS \>                 cd $env:WORKSPACE
  PS Code>              git clone https://github.com/Microsoft/vcpkg.git
  PS Code>              cd vcpkg
  PS Code\vcpkg>        .\bootstrap-vcpkg.bat

11) Open a Powershell with Administrator privileges and type

.. code-block:: bash

  PS \>                 cd $env:WORKSPACE
  PS Code>              cd vcpkg
  PS Code\vcpkg>        .\vcpkg integrate install

12) Install the Eigen library

.. code-block:: bash

  PS Code\vcpkg>        .\vcpkg install eigen3:x64-windows

13) Open a Powershell and build `plasticity` using the `build.ps1` script

.. code-block:: bash

  PS \>                 cd $env:WORKSPACE
  PS Code>              git clone https://github.com/Nico-Curti/plasticity
  PS Code>              cd plasticity
  PS Code\plasticity>   .\build.ps1 Release

.. _shut: https://github.com/Nico-Curti/Shut
