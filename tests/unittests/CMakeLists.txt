add_executable(basicNetworkTest BasicNet.cpp)
target_link_libraries(basicNetworkTest LINK_PUBLIC ${LinkLibs} gtest gtest_main)
add_test(basicNetworkTest ${GLOW_BINARY_DIR}/tests/basicNetworkTest)

add_executable(tensorsTest Tensors.cpp)
target_link_libraries(tensorsTest LINK_PUBLIC ${LinkLibs} gtest gtest_main)
add_test(tensorsTest ${GLOW_BINARY_DIR}/tests/tensorsTest)

add_executable(gradCheckTest GradCheck.cpp)
target_link_libraries(gradCheckTest LINK_PUBLIC ${LinkLibs} gtest gtest_main)
add_test(gradCheckTest ${GLOW_BINARY_DIR}/tests/gradCheckTest)
