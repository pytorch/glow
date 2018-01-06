
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOW_BINARY_DIR}/tests/)

include_directories(SYSTEM ${LLVM_SRC_ROOT}/utils/unittest/googletest/include)

add_subdirectory(unittests)
add_subdirectory(images)

