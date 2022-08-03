# If the pyx file is a C++ file, we should specify that here.

set_source_files_properties( "${CMAKE_CURRENT_SOURCE_DIR}/activations.pyx"            PROPERTIES CYTHON_IS_CXX TRUE )
set_source_files_properties( "${CMAKE_CURRENT_SOURCE_DIR}/bcm.pyx"                    PROPERTIES CYTHON_IS_CXX TRUE )
set_source_files_properties( "${CMAKE_CURRENT_SOURCE_DIR}/hopfield.pyx"               PROPERTIES CYTHON_IS_CXX TRUE )
set_source_files_properties( "${CMAKE_CURRENT_SOURCE_DIR}/weights_initialization.pyx" PROPERTIES CYTHON_IS_CXX TRUE )
set_source_files_properties( "${CMAKE_CURRENT_SOURCE_DIR}/update_args.pyx"            PROPERTIES CYTHON_IS_CXX TRUE )

# Include *.pxd directory

include_directories("${CMAKE_SOURCE_DIR}/plasticity/lib")
add_custom_target(PyPlasticity)

# Multi-file cython modules do not appear to be working at the moment.

cython_add_module(bcm "${CMAKE_CURRENT_SOURCE_DIR}/bcm.pyx")
target_link_libraries(bcm ${linked_libs} ${plasticitylib})
add_dependencies(bcm ${plasticitylib})

cython_add_module(hopfield "${CMAKE_CURRENT_SOURCE_DIR}/hopfield.pyx")
target_link_libraries(hopfield ${linked_libs} ${plasticitylib})
add_dependencies(hopfield ${plasticitylib})

cython_add_module(weights_initialization "${CMAKE_CURRENT_SOURCE_DIR}/weights_initialization.pyx")
target_link_libraries(weights_initialization ${linked_libs} ${plasticitylib})
add_dependencies(weights_initialization ${plasticitylib})

cython_add_module(update_args "${CMAKE_CURRENT_SOURCE_DIR}//update_args.pyx")
target_link_libraries(update_args ${linked_libs} ${plasticitylib})
add_dependencies(update_args ${plasticitylib})

add_dependencies(PyPlasticity bcm hopfield weights_initialization update_args)

# Installation of targets
install(TARGETS bcm                                                    DESTINATION "${CMAKE_SOURCE_DIR}/plasticity/lib")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/bcm.cpp"                    DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")
install(TARGETS hopfield                                               DESTINATION "${CMAKE_SOURCE_DIR}/plasticity/lib")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hopfield.cpp"               DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")
install(TARGETS weights_initialization                                 DESTINATION "${CMAKE_SOURCE_DIR}/plasticity/lib")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/weights_initialization.cpp" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")
install(TARGETS update_args                                            DESTINATION "${CMAKE_SOURCE_DIR}/plasticity/lib")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/update_args.cpp"            DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")


# setup installation

configure_file(
  "${CMAKE_SOURCE_DIR}/setup.py.in"
  "${CMAKE_SOURCE_DIR}/setup_.py" @ONLY)
configure_file(
  "${CMAKE_SOURCE_DIR}/plasticity/__version__.py.in"
  "${CMAKE_SOURCE_DIR}/plasticity/__version__.py" @ONLY)
configure_file(
  "${CMAKE_SOURCE_DIR}/setup.cfg.in"
  "${CMAKE_SOURCE_DIR}/setup.cfg" @ONLY)


add_custom_target(setup_plasticity ALL
  DEPENDS PyPlasticity
  COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_SOURCE_DIR}/setup_.py" develop --user
  #COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_SOURCE_DIR}/setup_.py" install --user
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Setup Python installation"
)
