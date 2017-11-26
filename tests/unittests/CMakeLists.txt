

add_executable(tensorsTest
                 Tensors.cpp)
target_link_libraries(tensorsTest
                      PRIVATE
                        Base
                        Support
                        gtest
                        gtest_main)
add_test(tensorsTest ${GLOW_BINARY_DIR}/tests/tensorsTest)

add_executable(IRgradCheckTest
                 IRGradCheck.cpp)
target_link_libraries(IRgradCheckTest
                      PRIVATE
                        Backends
                        ExecutionEngine
                        IR
                        gtest
                        gtest_main)
add_test(IRgradCheckTest ${GLOW_BINARY_DIR}/tests/IRgradCheckTest)

add_executable(IRTest
                 IRTest.cpp)
target_link_libraries(IRTest
                      PRIVATE
                        Backends
                        IR
                        gtest
                        gtest_main)
add_test(IRTest ${GLOW_BINARY_DIR}/tests/IRTest)

add_executable(InterpreterTest
                 InterpreterTest.cpp)
target_link_libraries(InterpreterTest
                      PRIVATE
                        Backends
                        IR
                        ExecutionEngine
                        gtest
                        gtest_main)
add_test(InterpreterTest ${GLOW_BINARY_DIR}/tests/InterpreterTest)

add_executable(GraphTest
                 GraphTest.cpp)
target_link_libraries(GraphTest
                      PRIVATE
                        Graph
                        IR
                        Backends
                        gtest
                        gtest_main)
add_test(GraphTest ${GLOW_BINARY_DIR}/tests/GraphTest)

add_executable(GraphGradTest
                 GraphAutoGrad.cpp)
target_link_libraries(GraphGradTest
                      PRIVATE
                        Graph
                        IR
                        Backends
                        ExecutionEngine
                        gtest
                        gtest_main)
add_test(GraphGradTest ${GLOW_BINARY_DIR}/tests/GraphGradTest)


add_executable(GraphOptz
                 GraphOpt.cpp)
target_link_libraries(GraphOptz
                      PRIVATE
                        Graph
                        IR
                        Backends
                        gtest
                        gtest_main)
add_test(GraphOptz ${GLOW_BINARY_DIR}/tests/GraphOptz)

if(OpenCL_FOUND)
add_executable(OCLTest
                 OCLTest.cpp)
target_link_libraries(OCLTest
                      PRIVATE
                        Backends
                        IR
                        ExecutionEngine
                        gtest
                        gtest_main)
add_test(OCLTest ${GLOW_BINARY_DIR}/tests/OCLTest)
endif()

add_executable(MemoryAllocatorTest
                MemoryAllocatorTest.cpp)
target_link_libraries(MemoryAllocatorTest
                      PRIVATE
                        Backends
                        IR
                        gtest
                        gtest_main)
add_test(MemoryAllocatorTest ${GLOW_BINARY_DIR}/tests/MemoryAllocatorTest)
