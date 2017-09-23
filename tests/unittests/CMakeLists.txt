
add_executable(basicNetworkTest
                 BasicNet.cpp)
target_link_libraries(basicNetworkTest
                      PRIVATE
                        Network
                        gtest
                        gtest_main)
add_test(basicNetworkTest ${GLOW_BINARY_DIR}/tests/basicNetworkTest)

add_executable(tensorsTest
                 Tensors.cpp)
target_link_libraries(tensorsTest
                      PRIVATE
                        Support
                        gtest
                        gtest_main)
add_test(tensorsTest ${GLOW_BINARY_DIR}/tests/tensorsTest)

add_executable(gradCheckTest
                 GradCheck.cpp)
target_link_libraries(gradCheckTest
                      PRIVATE
                        Network
                        gtest
                        gtest_main)
add_test(gradCheckTest ${GLOW_BINARY_DIR}/tests/gradCheckTest)


add_executable(IRgradCheckTest
                 IRGradCheck.cpp)
target_link_libraries(IRgradCheckTest
                      PRIVATE
                        Network
                        Interpreter
                        IR
                        gtest
                        gtest_main)
add_test(IRgradCheckTest ${GLOW_BINARY_DIR}/tests/IRgradCheckTest)

add_executable(IRTest
                 IRTest.cpp)
target_link_libraries(IRTest
                      PRIVATE
                        Network
                        Interpreter
                        IR
                        gtest
                        gtest_main)
add_test(IRTest ${GLOW_BINARY_DIR}/tests/IRTest)

add_executable(InterpreterTest
                 InterpreterTest.cpp)
target_link_libraries(InterpreterTest
                      PRIVATE
                        Interpreter
                        Network
                        IR
                        gtest
                        gtest_main)
add_test(InterpreterTest ${GLOW_BINARY_DIR}/tests/InterpreterTest)
