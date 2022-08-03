cmake_minimum_required (VERSION 3.9.0)

# plasticity Version
set (PLASTICITY_MAJOR    1)
set (PLASTICITY_MINOR    0)
set (PLASTICITY_REVISION 2)
set (PLASTICITY_VERSION ${PLASTICITY_MAJOR}.${PLASTICITY_MINOR}.${PLASTICITY_REVISION})

set (CMAKE_PROJECT_HOMEPAGE_URL "https://github.com/Nico-Curti/plasticity")
set (CMAKE_PROJECT_DESCRIPTION "Unsupervised Neural Networks with biological-inspired learning rules")

message (STATUS "plasticity VERSION: ${PLASTICITY_VERSION}")

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
  message(STATUS "VCPKG found: $ENV{VCPKG_ROOT}")
  message(STATUS "Using VCPKG integration")
  message(STATUS "VCPKG_MANIFEST_FEATURES: ${VCPKG_MANIFEST_FEATURES}")
  if(DEFINED ENV{VCPKG_DEFAULT_TRIPLET})
    message(STATUS "Setting default vcpkg target triplet to $ENV{VCPKG_DEFAULT_TRIPLET}")
    set(VCPKG_TARGET_TRIPLET $ENV{VCPKG_DEFAULT_TRIPLET})
  endif()
endif()

project (plasticity LANGUAGES CXX VERSION ${PLASTICITY_VERSION} DESCRIPTION "${CMAKE_PROJECT_DESCRIPTION}")

enable_language(CXX)

if ( ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" AND ${CMAKE_CXX_COMPILER_VERSION} LESS 7) OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND ${CMAKE_CXX_COMPILER_VERSION} LESS_EQUAL 4) )
  set (CMAKE_CXX_STANDARD 14)
else ()
  set (CMAKE_CXX_STANDARD 17)
endif()

add_definitions (-DMAJOR=${PLASTICITY_MAJOR} -DMINOR=${PLASTICITY_MINOR} -DREVISION=${PLASTICITY_REVISION})

#################################################################
#                         COMPILE OPTIONS                       #
#################################################################

option (OMP        "Enable OpenMP                support" OFF)
option (BUILD_TEST "Enable tests build           support" OFF)
option (PYWRAP     "Enable Python wrap compilation      " OFF)
option (BUILD_DOCS "Enable Documentaion builid   support" OFF)
option (VERBOSE    "Enable verbosity             support" OFF)
option (VIEW       "Enable OpenCV                support" OFF)

#################################################################
#                         SETTING VARIABLES                     #
#################################################################

set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules/" ${CMAKE_MODULE_PATH})

if ( NOT APPLE )
  set (CMAKE_SKIP_BUILD_RPATH             FALSE )
  set (CMAKE_BUILD_WITH_INSTALL_RPATH     FALSE )
  set (CMAKE_INSTALL_RPATH_USE_LINK_PATH  TRUE  )
endif()

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "Install prefix" FORCE)
endif()

if ( CMAKE_COMPILER_IS_GNUCXX )
  add_compile_options (-Wall -Wextra -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-narrowing -Wpedantic)
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    string (REGEX REPLACE "-O[0123]" "-Og" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
    set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG" )
    set (CMAKE_CXX_FLAGS_RELEASE "")
  elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    string (REGEX REPLACE "-O3" "-Ofast" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG" )
    set (CMAKE_CXX_FLAGS_DEBUG "")
  endif()
  #list (APPEND linked_libs stdc++fs)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  add_compile_options(-Wno-deprecated -Wno-writable-strings)
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    string (REGEX REPLACE "-O0" "-Og" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
    set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG" )
  elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    # there are some issues in the clang symbols with Ofast
    #string (REGEX REPLACE "-O3" "-Ofast" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG" )
  endif()
endif()

if (MSVC)
  set (CMAKE_CXX_FLAGS "/wd4013 /wd4018 /wd4028 /wd4047 /wd4068 /wd4090 /wd4101 /wd4113 /wd4133 /wd4190 /wd4244 /wd4267 /wd4305 /wd4477 /wd4996 /wd4819 /fp:fast ${CMAKE_CXX_FLAGS}")
  string (REGEX REPLACE "/O2" "/Ox" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  add_definitions (-D_CRT_RAND_S)
  add_definitions (-DNOMINMAX)
  #add_definitions (-D_USE_MATH_DEFINES)
  add_definitions (-D_CRT_SECURE_NO_DEPRECATE -D_SCL_SECURE_NO_WARNINGS)
  set (CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()


include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if (COMPILER_SUPPORTS_MARCH_NATIVE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()


#################################################################
#                         PARSE OPTIONS                         #
#################################################################

find_package (Eigen3 REQUIRED NO_MODULE)
if (EIGEN3_FOUND)
  message(STATUS "Eigen found: version ${EIGEN3_VERSION_STRING}")

  if (${EIGEN3_VERSION_STRING} VERSION_LESS "3.3.9")
    message (STATUS "WARNING! Slicing optimization disabled (available from Eigen version >= 3.3.9)")
  endif ()

  include_directories(${EIGEN3_INCLUDE_DIR})
else ()
  message(FATAL_ERROR "Eigen not found")
endif ()

if (OMP)
  find_package (OpenMP REQUIRED)
  if (OPENMP_FOUND)
    message (STATUS "OpenMP found")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    if (APPLE)
      list (APPEND linked_libs OpenMP::OpenMP_CXX)
    endif ()
  endif ()
else ()
  message (STATUS "OpenMP - disabled")
  add_definitions(-DEIGEN_DONT_PARALLELIZE)
endif ()

if (VERBOSE)
  add_definitions (-D__verbose__)
endif ()

if (VIEW)
  find_package (OpenCV REQUIRED COMPONENTS core highgui)
  message(STATUS "OpenCV found: version ${OpenCV_VERSION}")
  include_directories (${OpenCV_INCLUDE_DIRS})
  list(APPEND linked_libs "opencv_core")
  list(APPEND linked_libs "opencv_highgui")
  add_definitions (-D__view__)
endif ()

#################################################################
#                         SETTING DIRECTORIES                   #
#################################################################

set(INSTALL_BIN_DIR "${CMAKE_CURRENT_LIST_DIR}/bin" CACHE PATH "Path where exe and dll will be installed")
set(INSTALL_LIB_DIR "${CMAKE_CURRENT_LIST_DIR}/lib" CACHE PATH "Path where lib will be installed")

set(INSTALL_INCLUDE_DIR  "include/plasticity"  CACHE PATH "Path where headers will be installed")
set(INSTALL_CMAKE_DIR    "share/plasticity"    CACHE PATH "Path where cmake configs will be installed")

set(CMAKE_DEBUG_POSTFIX d)

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/include/version.h.in"
  "${CMAKE_CURRENT_LIST_DIR}/include/version.h" @ONLY)

file(GLOB PLASTICITY_SRC     "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp"  )
file(GLOB PLASTICITY_HEADER  "${CMAKE_CURRENT_LIST_DIR}/include/*.h")
file(GLOB PLASTICITY_HPP     "${CMAKE_CURRENT_LIST_DIR}/hpp/*.hpp"  )

include_directories("${CMAKE_CURRENT_LIST_DIR}/include")
include_directories("${CMAKE_CURRENT_LIST_DIR}/hpp")

if (PYWRAP)
  include( UseCython )
  find_package (Python3 REQUIRED COMPONENTS Interpreter)
  find_package(NumPy REQUIRED)
  include_directories(${NumPy_INCLUDE_DIRS})
  add_definitions (-DNPY_NO_DEPRECATED_API)
endif()

set(plasticitylib plasticity)

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/plasticity.pc.in"
  "${CMAKE_CURRENT_LIST_DIR}/plasticity.pc" @ONLY)
message(STATUS "Pkg-config generated")

# allow the export of LD_LIBRARY_PATH env variable
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Make relative paths absolute (needed later on)
foreach (p LIB BIN INCLUDE CMAKE)
  set (var INSTALL_${p}_DIR)
  if (NOT IS_ABSOLUTE "${${var}}")
    set (FULLPATH_${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

#################################################################
#                          SUMMARY                              #
#################################################################

message(STATUS ""                                                                    )
message(STATUS "================ Plasticity configuration Summary ==============="   )
message(STATUS "   Plasticity version: ${PLASTICITY_VERSION}"                        )
message(STATUS ""                                                                    )
message(STATUS "   Build type : ${CMAKE_BUILD_TYPE}"                                 )
message(STATUS "   C++ :"                                                            )
message(STATUS "      C++ Compiler : ${CMAKE_CXX_COMPILER}"                          )
message(STATUS "      C++ flags    :"                                                )
foreach(FLAG ${CMAKE_CXX_FLAGS_LIST})
  message(STATUS "                    * ${FLAG}"                                     )
endforeach(FLAG)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "      C++ Debug flags:"                                            )
  foreach(FLAG ${CMAKE_CXX_FLAGS_DEBUG})
    message(STATUS "                    * ${FLAG}"                                   )
  endforeach(FLAG)
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "      C++ Release flags  :"                                        )
  foreach(FLAG ${CMAKE_CXX_FLAGS_RELEASE})
    message(STATUS "                    * ${FLAG}"                                   )
  endforeach(FLAG)
endif()
message(STATUS "      Linker flags : "                                               )
foreach(FLAG ${linked_libs})
  message(STATUS "                    * ${FLAG}"                                     )
endforeach(FLAG)
message(STATUS ""                                                                    )
message(STATUS "   OpenMP support : ${OMP}"                                          )
message(STATUS "   Enable build testing : ${BUILD_TEST}"                             )
message(STATUS "   Enable Progress bar during training : ${VERBOSE}"                 )
message(STATUS "   Enable OpenCV support : ${VIEW}"                                  )
message(STATUS "   Compile Pythonize version : ${PYWRAP}"                            )
message(STATUS "   Documentation support : ${BUILD_DOCS}"                            )
message(STATUS ""                                                                    )


#################################################################
#                         MAIN RULES                            #
#################################################################

add_library(${plasticitylib} SHARED ${PLASTICITY_SRC} ${PLASTICITY_HEADER} ${PLASTICITY_HPP})
set_property(TARGET ${plasticitylib} PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(${plasticitylib}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hpp>
  $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}>)
target_link_libraries(${plasticitylib} ${linked_libs})
#target_compile_definitions(${plasticitylib} PRIVATE -D__verbose__)

list (APPEND ALL_HEADER ${PLASTICITY_HEADER})
list (APPEND ALL_HEADER ${PLASTICITY_HPP})
set_target_properties(${plasticitylib} PROPERTIES PUBLIC_HEADER "${ALL_HEADER}")

add_executable(example_bcm_mnist "${CMAKE_CURRENT_LIST_DIR}/example/run_bcm_mnist.cpp")
target_link_libraries(example_bcm_mnist ${linked_libs} ${plasticitylib})

add_executable(example_bcm_cifar10 "${CMAKE_CURRENT_LIST_DIR}/example/run_bcm_cifar10.cpp")
target_link_libraries(example_bcm_cifar10 ${linked_libs} ${plasticitylib})

add_executable(example_hopfield_mnist "${CMAKE_CURRENT_LIST_DIR}/example/run_hopfield_mnist.cpp")
target_link_libraries(example_hopfield_mnist ${linked_libs} ${plasticitylib})

add_executable(example_hopfield_cifar10 "${CMAKE_CURRENT_LIST_DIR}/example/run_hopfield_cifar10.cpp")
target_link_libraries(example_hopfield_cifar10 ${linked_libs} ${plasticitylib})

add_executable(example_mnist "${CMAKE_CURRENT_LIST_DIR}/example/run_mnist.cpp")
target_link_libraries(example_mnist ${linked_libs} ${plasticitylib})

add_executable(example_cifar10 "${CMAKE_CURRENT_LIST_DIR}/example/run_cifar10.cpp")
target_link_libraries(example_cifar10 ${linked_libs} ${plasticitylib})

add_executable(plasticity_infos "${CMAKE_CURRENT_LIST_DIR}/example/plasticity_infos.cpp")
target_link_libraries(plasticity_infos ${linked_libs} ${plasticitylib})


if (BUILD_TEST)
  add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/testing")
endif()

if (PYWRAP)
  add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/plasticity/source")
endif()

# This must be the latest subdirectory included!!
if (BUILD_DOCS)
  add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/docs/")
endif ()

#################################################################
#                          INSTALLERS                           #
#################################################################

install(TARGETS ${plasticitylib} EXPORT plasticityTargets
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
  PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}"
  COMPONENT dev
)

install(TARGETS example_bcm_mnist        DESTINATION "${INSTALL_BIN_DIR}")
install(TARGETS example_bcm_cifar10      DESTINATION "${INSTALL_BIN_DIR}")
install(TARGETS example_hopfield_mnist   DESTINATION "${INSTALL_BIN_DIR}")
install(TARGETS example_hopfield_cifar10 DESTINATION "${INSTALL_BIN_DIR}")
install(TARGETS example_mnist            DESTINATION "${INSTALL_BIN_DIR}")
install(TARGETS example_cifar10          DESTINATION "${INSTALL_BIN_DIR}")
install(TARGETS plasticity_infos         DESTINATION "${INSTALL_BIN_DIR}")

install(EXPORT plasticityTargets
  FILE plasticityTargets.cmake
  NAMESPACE plasticity::
  DESTINATION "${INSTALL_CMAKE_DIR}"
)


# Export the package for use from the build-tree (this registers the build-tree with a global CMake-registry)
export(PACKAGE plasticity)

# Create the plasticityConfig.cmake
# First of all we compute the relative path between the cmake config file and the include path
file(RELATIVE_PATH REL_INCLUDE_DIR "${FULLPATH_INSTALL_CMAKE_DIR}" "${FULLPATH_INSTALL_INCLUDE_DIR}")
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
configure_file(plasticityConfig.cmake.in "${PROJECT_BINARY_DIR}/plasticityConfig.cmake" @ONLY)
set(CONF_INCLUDE_DIRS "\${plasticity_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(plasticityConfig.cmake.in "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/plasticityConfig.cmake" @ONLY)

# Create the plasticityConfigVersion.cmake
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_BINARY_DIR}/plasticityConfigVersion.cmake"
  COMPATIBILITY SameMajorVersion
)

install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/plasticityConfig.cmake"
  "${PROJECT_BINARY_DIR}/plasticityConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}"
)
