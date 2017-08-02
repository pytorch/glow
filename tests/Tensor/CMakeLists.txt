add_executable(tensorTest tensor.cpp)
target_link_libraries(tensorTest LINK_PUBLIC ${LinkLibs} gtest gtest_main)
add_test(tensorTest tensorTest)

