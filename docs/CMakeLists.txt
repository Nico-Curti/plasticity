
# build C++ APIs with Doxygen

find_package(Doxygen)

configure_file(
  "${CMAKE_SOURCE_DIR}/docs/Doxyfile.in"
  "${CMAKE_SOURCE_DIR}/docs/Doxyfile" @ONLY
)
configure_file(
  "${CMAKE_SOURCE_DIR}/docs/source/conf.py.in"
  "${CMAKE_SOURCE_DIR}/docs/source/conf.py" @ONLY
)


add_custom_target( cpp_docs ALL
    DEPENDS ${plasticitylib}
    COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_SOURCE_DIR}/docs/Doxyfile"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Generating C++ API documentation with Doxygen"
    VERBATIM )


if (PYWRAP)

  # build Python APIs with Sphinx

  find_package(Sphinx REQUIRED)

  set(SPHINX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/source/")
  set(SPHINX_BUILD "${CMAKE_SOURCE_DIR}/docs/build")

  add_custom_target(py_docs ALL
                    DEPENDS setup_plasticity cpp_docs
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                    COMMENT "Generating Python API documentation with Sphinx")

  # Workaround for the lack of post_install steps.
  install(CODE "set(CMAKE_BINARY_DIR \"${CMAKE_BINARY_DIR}\")")
  install(CODE "set(SPHINX_SOURCE \"${SPHINX_SOURCE}\")")
  install(CODE "set(SPHINX_BUILD \"${SPHINX_BUILD}\")")
  install(CODE "set(SPHINX_EXECUTABLE \"${SPHINX_EXECUTABLE}\")")
  install(CODE "set(DOXYGEN_OUTPUT_DIR \"${DOXYGEN_OUTPUT_DIR}\")")

  install(SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/RunSphinx.cmake")

endif ()
