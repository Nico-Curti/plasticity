# If the pyx file is a C++ file, we should specify that here.

set(PYC_SRC_DIR    ${PYC_DIR}/source       CACHE PATH "Path where find pyx files" )
set(PYC_INC_DIR    ${PYC_DIR}/lib          CACHE PATH "Path where find pxd files" )

set_source_files_properties( ${PYC_SRC_DIR}/activations.pyx PROPERTIES CYTHON_IS_CXX TRUE )
set_source_files_properties( ${PYC_SRC_DIR}/bcm.pyx PROPERTIES CYTHON_IS_CXX TRUE )
set_source_files_properties( ${PYC_SRC_DIR}/hopfield.pyx PROPERTIES CYTHON_IS_CXX TRUE )
set_source_files_properties( ${PYC_SRC_DIR}/weights_initialization.pyx PROPERTIES CYTHON_IS_CXX TRUE )
set_source_files_properties( ${PYC_SRC_DIR}/update_args.pyx PROPERTIES CYTHON_IS_CXX TRUE )

# Include *.pxd directory

include_directories(${PYC_INC_DIR})
add_custom_target(PyPlasticity)

# Multi-file cython modules do not appear to be working at the moment.

cython_add_module( bcm  ${PYC_SRC_DIR}/bcm.pyx
                                                                ${SRC_DIR}/activations.cpp
                                                                ${SRC_DIR}/bcm.cpp
                                                                ${SRC_DIR}/base.cpp
                                                                ${SRC_DIR}/fmath.cpp
                                                                ${SRC_DIR}/optimizer.cpp
                                                                ${SRC_DIR}/weights.cpp
                                                                ${SRC_DIR}/utils.cpp
                                                                )
target_link_libraries(bcm ${linked_libs})

cython_add_module( hopfield  ${PYC_SRC_DIR}/hopfield.pyx
                                                                ${SRC_DIR}/activations.cpp
                                                                ${SRC_DIR}/base.cpp
                                                                ${SRC_DIR}/fmath.cpp
                                                                ${SRC_DIR}/hopfield.cpp
                                                                ${SRC_DIR}/optimizer.cpp
                                                                ${SRC_DIR}/weights.cpp
                                                                ${SRC_DIR}/utils.cpp
                                                                )
target_link_libraries(hopfield ${linked_libs})

cython_add_module( weights_initialization  ${PYC_SRC_DIR}/weights_initialization.pyx
                                                                ${SRC_DIR}/fmath.cpp
                                                                ${SRC_DIR}/weights.cpp
                                                                ${SRC_DIR}/utils.cpp
                                                                )
target_link_libraries(weights_initialization ${linked_libs})

cython_add_module( update_args  ${PYC_SRC_DIR}/update_args.pyx
                                                                ${SRC_DIR}/fmath.cpp
                                                                ${SRC_DIR}/optimizer.cpp
                                                                ${SRC_DIR}/utils.cpp
                                                                )
target_link_libraries(update_args ${linked_libs})

add_dependencies(PyPlasticity bcm hopfield weights_initialization update_args)

# Installation of targets
install(TARGETS bcm                                                   DESTINATION ${PYC_INC_DIR}/)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/bcm.cpp                     DESTINATION ${PYC_SRC_DIR})
install(TARGETS hopfield                                              DESTINATION ${PYC_INC_DIR}/)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hopfield.cpp                DESTINATION ${PYC_SRC_DIR})
install(TARGETS weights_initialization                                DESTINATION ${PYC_INC_DIR}/)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/weights_initialization.cpp  DESTINATION ${PYC_SRC_DIR})
install(TARGETS update_args                                           DESTINATION ${PYC_INC_DIR}/)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/update_args.cpp             DESTINATION ${PYC_SRC_DIR})


# setup installation

set(SETUP_PY_IN "${CMAKE_SOURCE_DIR}/setup.py.in")
set(SETUP_PY    "${CMAKE_SOURCE_DIR}/setup_.py")

configure_file(${SETUP_PY_IN} ${SETUP_PY} @ONLY)
configure_file(${PYC_DIR}/__version__.py.in ${PYC_DIR}/__version__.py @ONLY)

add_custom_target(setup_plasticity ALL
                  DEPENDS PyPlasticity
                  COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} develop --user
                  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                  COMMENT "Setup Python installation")
