include_directories(${GTEST_INCLUDE_DIRS})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOW_BINARY_DIR}/tests/)
add_subdirectory(googletest/)
add_subdirectory(unittests)

