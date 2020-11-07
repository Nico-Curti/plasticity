# Test files


add_executable(test_bcm ${TEST_DIR}/test_bcm.cpp ${SRC} ${HEADER} ${HPP})
target_link_libraries(test_bcm ${linked_libs})
target_compile_definitions(test_bcm PRIVATE -D__testing__)

add_executable(test_hopfield ${TEST_DIR}/test_hopfield.cpp ${SRC} ${HEADER} ${HPP})
target_link_libraries(test_hopfield ${linked_libs})
target_compile_definitions(test_hopfield PRIVATE -D__testing__)


# Installation of targets

install(TARGETS test_bcm           DESTINATION ${OUT_TEST})
install(TARGETS test_hopfield      DESTINATION ${OUT_TEST})
