
execute_process(COMMAND ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH="${CMAKE_BINARY_DIR}" ${SPHINX_EXECUTABLE} -b html
                        # Tell Breathe where to find the Doxygen output
                        -Dbreathe_projects.CatCutifier=${DOXYGEN_OUTPUT_DIR}
                        ${SPHINX_SOURCE} ${SPHINX_BUILD})
