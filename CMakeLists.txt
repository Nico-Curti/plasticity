cmake_minimum_required (VERSION 3.9.0)

# plasticity Version
set (PLASTICITY_MAJOR    1)
set (PLASTICITY_MINOR    0)
set (PLASTICITY_REVISION 1)
set (PLASTICITY_VERSION ${PLASTICITY_MAJOR}.${PLASTICITY_MINOR}.${PLASTICITY_REVISION})

project (plasticity LANGUAGES CXX VERSION ${PLASTICITY_VERSION} DESCRIPTION "Encoding models")

if ( ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" AND ${CMAKE_CXX_COMPILER_VERSION} LESS 7) OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND ${CMAKE_CXX_COMPILER_VERSION} LESS_EQUAL 4) )
  set (CMAKE_CXX_STANDARD 14)
else ()
  set (CMAKE_CXX_STANDARD 17)
endif()

add_definitions (-DMAJOR=${PLASTICITY_MAJOR} -DMINOR=${PLASTICITY_MINOR} -DREVISION=${PLASTICITY_REVISION})

#################################################################
#                         COMPILE OPTIONS                       #
#################################################################

option (BUILD_TEST "Enable tests build           support" OFF)
option (PYWRAP     "Enable Python wrap compilation      " OFF)
option (BUILD_DOCS "Enable Documentaion builid   support" OFF)
option (VERBOSE    "Enable verbosity             support" ON)
option (VIEW       "Enable OpenCV                support" OFF)

#################################################################
#                         SETTING VARIABLES                     #
#################################################################

set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/" ${CMAKE_MODULE_PATH})

if ( NOT APPLE )
  set (CMAKE_SKIP_BUILD_RPATH             FALSE )
  set (CMAKE_BUILD_WITH_INSTALL_RPATH     FALSE )
  set (CMAKE_INSTALL_RPATH_USE_LINK_PATH  TRUE  )
endif()

# make sure that the default is a RELEASE
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
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

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Install prefix" FORCE)
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

set(SRC_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/src                CACHE PATH "Path where find cpp files"                        )
set(INC_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/include            CACHE PATH "Path where find header files"                     )
set(HPP_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/hpp                CACHE PATH "Path where find hpp files"                        )
set(EXAMPLE  ${CMAKE_CURRENT_SOURCE_DIR}/example            CACHE PATH "Path where find main files"                       )
set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/testing            CACHE PATH "Path where find test files"                       )
set(PYC_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/plasticity         CACHE PATH "Path where find cython files"                     )         # cython directory
set(LIB_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/lib                CACHE PATH "Path where lib will be installed"            FORCE)
set(INST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/share/plasticity   CACHE PATH "Path where cmake configs will be installed"  FORCE)
set(OUT_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/bin                CACHE PATH "Path where outputs will be installed"        FORCE)
set(OUT_TEST ${TEST_DIR}/bin                                CACHE PATH "Path where test outputs will be installed"   FORCE)

set(INSTALL_INCLUDE_DIR  "${CMAKE_CURRENT_LIST_DIR}/share/include/plasticity"  CACHE PATH "Path where headers will be installed")
set(INSTALL_CMAKE_DIR    "${CMAKE_CURRENT_LIST_DIR}/share/plasticity"          CACHE PATH "Path where cmake configs will be installed")

set(CMAKE_DEBUG_POSTFIX d)

set(plasticitylib plasticity)

file(GLOB SRC           "${SRC_DIR}/*.cpp" )
file(GLOB HEADER        "${INC_DIR}/*.h"   )
file(GLOB HPP           "${INC_DIR}/*.hpp" )

include_directories(${INC_DIR} ${HPP_DIR} ${TEST_DIR})

if (PYWRAP)
  include( UseCython )
  find_package (Python REQUIRED COMPONENTS Interpreter NumPy)
  include_directories(${Python_NumPy_INCLUDE_DIRS})
  add_definitions (-DNPY_NO_DEPRECATED_API)
endif()

if (BUILD_DOCS)
  add_subdirectory(${CMAKE_SOURCE_DIR}/docs/)
endif ()

set(plasticity_INSTALL_INCLUDE_DIR ${INSTALL_INCLUDE_DIR})

configure_file(${CMAKE_SOURCE_DIR}/plasticity.pc.in ${CMAKE_SOURCE_DIR}/plasticity.pc @ONLY)
configure_file(${PYC_DIR}/__version__.py.in ${PYC_DIR}/__version__.py @ONLY)
configure_file(${INC_DIR}/version.h.in ${INC_DIR}/version.h @ONLY)
message(STATUS "Pkg-config generated")

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
message(STATUS "   Enable build testing : ${BUILD_TEST}"                             )
message(STATUS "   Enable Progress bar during training : ${VERBOSE}"                 )
message(STATUS "   Enable OpenCV support : ${VIEW}"                                  )
message(STATUS "   Compile Pythonize version : ${PYWRAP}"                            )
message(STATUS "   Documentation support : ${BUILD_DOCS}"                            )
message(STATUS ""                                                                    )


#################################################################
#                         MAIN RULES                            #
#################################################################

if (PYWRAP)
  add_subdirectory(${PYC_DIR}/source)
endif()

add_library(${plasticitylib} SHARED ${SRC} ${HEADER} ${HPP})
set_property(TARGET ${plasticitylib} PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(${plasticitylib} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include> $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/hpp> $<INSTALL_INTERFACE:${rFBP_INSTALL_INCLUDE_DIR}>)
target_link_libraries(${plasticitylib} ${linked_libs})
#target_compile_definitions(${plasticitylib} PRIVATE -D__verbose__)

list (APPEND ALL_HEADER ${HEADER})
list (APPEND ALL_HEADER ${HPP})
set_target_properties(${plasticitylib} PROPERTIES PUBLIC_HEADER "${ALL_HEADER}")

add_executable(example_bcm ${EXAMPLE}/run_bcm.cpp)
target_link_libraries(example_bcm ${linked_libs} ${plasticitylib})

add_executable(example_hopfield ${EXAMPLE}/run_hopfield.cpp)
target_link_libraries(example_hopfield ${linked_libs} ${plasticitylib})

add_executable(example_mnist ${EXAMPLE}/run_mnist.cpp)
target_link_libraries(example_mnist ${linked_libs} ${plasticitylib})

add_executable(plasticity_infos ${EXAMPLE}/plasticity_infos.cpp)
target_link_libraries(plasticity_infos ${linked_libs} ${plasticitylib})


if (BUILD_TEST)
  add_subdirectory(${TEST_DIR})
endif()

#################################################################
#                          INSTALLERS                           #
#################################################################

install(TARGETS ${plasticitylib}    EXPORT plasticityTargets
                                    RUNTIME DESTINATION ${OUT_DIR}
                                    LIBRARY DESTINATION ${LIB_DIR}
                                    ARCHIVE DESTINATION ${LIB_DIR}
                                    PUBLIC_HEADER DESTINATION ${INSTALL_INCLUDE_DIR}
                                    COMPONENT dev
                                    )

install(TARGETS example_bcm        DESTINATION ${OUT_DIR})
install(TARGETS example_hopfield   DESTINATION ${OUT_DIR})
install(TARGETS example_mnist      DESTINATION ${OUT_DIR})
install(TARGETS plasticity_infos   DESTINATION ${OUT_DIR})
