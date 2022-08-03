#!/bin/bash

red=$(tput setaf 196)
green=$(tput setaf 10)
reset=$(tput sgr0)

# get the directory of the current bash script (independently by the execution path)
scriptdir=$(dirname $(readlink /proc/$$/fd/255))
pushd $scriptdir > /dev/null

# $1 debug or release
build_type=$1
compiler=$(echo "${CXX##*/}")
number_of_build_workers=$(grep -c ^processor /proc/cpuinfo)

other_cmake_flags="${@:2}"

elif [ "$build_type" == "Release" ] || [ "$build_type" == "release" ]; then
  echo "${green}Building Release project${reset}"
  build_type=Release
  #rm -rf build_release
  mkdir -p build_release
  cd build_release

  cmake .. -DCMAKE_BUILD_TYPE=$build_type $other_cmake_flags
  cmake --build . --target install --parallel $number_of_build_workers
  cd ..

elif [ "$build_type" == "Debug" ] || [ "$build_type" == "debug" ]; then
  echo "${green}Building Debug project${reset}"
  build_type=Debug
  #rm -rf build_debug
  mkdir -p build_debug
  cd build_debug

  cmake .. -DCMAKE_BUILD_TYPE=$build_type $other_cmake_flags
  cmake --build . --target install --parallel $number_of_build_workers
  cd ..

else
  echo "${red}Unknown build type - Allowed only [Debug, Release]${reset}"
  exit 1

fi

popd > /dev/null
