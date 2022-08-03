# Test files

add_executable(test_bcm "${CMAKE_CURRENT_SOURCE_DIR}/test_bcm.cpp")
target_link_libraries(test_bcm ${linked_libs} ${plasticitylib})
target_compile_definitions(test_bcm PRIVATE -D__testing__)
target_include_directories(test_bcm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_hopfield "${CMAKE_CURRENT_SOURCE_DIR}/test_hopfield.cpp")
target_link_libraries(test_hopfield ${linked_libs} ${plasticitylib})
target_compile_definitions(test_hopfield PRIVATE -D__testing__)
target_include_directories(test_hopfield PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

# Installation of targets

install(TARGETS test_bcm      DESTINATION "${INSTALL_BIN_DIR}")
install(TARGETS test_hopfield DESTINATION "${INSTALL_BIN_DIR}")
