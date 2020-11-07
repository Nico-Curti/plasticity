# If the pyx file is a C++ file, we should specify that here.

set(PYC_SRC_DIR    ${PYC_DIR}/source       CACHE PATH "Path where find pyx files" )
set(PYC_INC_DIR    ${PYC_DIR}/lib          CACHE PATH "Path where find pxd files" )

set_source_files_properties( ${PYC_SRC_DIR}/activations.pyx PROPERTIES CYTHON_IS_CXX TRUE )
set_source_files_properties( ${PYC_SRC_DIR}/bcm.pyx PROPERTIES CYTHON_IS_CXX TRUE )

# Include *.pxd directory

include_directories(${PYC_INC_DIR})
add_custom_target(PyPlasticity)

# Multi-file cython modules do not appear to be working at the moment.

cython_add_module( bcm  ${PYC_SRC_DIR}/bcm.pyx
                                                                ${SRC_DIR}/activations.cpp
                                                                ${SRC_DIR}/bcm.cpp
                                                                ${SRC_DIR}/base.cpp
                                                                ${SRC_DIR}/fmath.cpp
                                                                ${SRC_DIR}/utils.cpp
                                                                )
target_link_libraries(bcm ${linked_libs})

cython_add_module( hopfield  ${PYC_SRC_DIR}/hopfield.pyx
                                                                ${SRC_DIR}/activations.cpp
                                                                ${SRC_DIR}/base.cpp
                                                                ${SRC_DIR}/fmath.cpp
                                                                ${SRC_DIR}/hopfield.cpp
                                                                ${SRC_DIR}/utils.cpp
                                                                )
target_link_libraries(hopfield ${linked_libs})

add_dependencies(PyPlasticity bcm) # hopfield

# Installation of targets
install(TARGETS bcm                                        DESTINATION ${PYC_INC_DIR}/)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/bcm.cpp          DESTINATION ${PYC_SRC_DIR})
install(TARGETS hopfield                                DESTINATION ${PYC_INC_DIR}/)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hopfield.cpp  DESTINATION ${PYC_SRC_DIR})


# setup installation

#set(SETUP_PY "${CMAKE_SOURCE_DIR}/setup.py")
#install(CODE "EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} develop --user)")
