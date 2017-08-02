add_executable(networkTest network.cpp)
target_link_libraries(networkTest LINK_PUBLIC ${LinkLibs} gtest gtest_main)
add_test(networkTest networkTest)

