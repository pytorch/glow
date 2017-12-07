

add_executable(tensorsTest
                 Tensors.cpp)
target_link_libraries(tensorsTest
                      PRIVATE
                        Base
                        Support
                        gtest
                        gtest_main)
add_test(tensorsTest ${GLOW_BINARY_DIR}/tests/tensorsTest)

add_executable(gradCheckTest
                 GradCheck.cpp)
target_link_libraries(gradCheckTest
                      PRIVATE
                        Backends
                        ExecutionEngine
                        IR
                        gtest
                        gtest_main)
add_test(gradCheckTest ${GLOW_BINARY_DIR}/tests/gradCheckTest)

add_executable(IROptTest
                 IROpt.cpp)
target_link_libraries(IROptTest
                      PRIVATE
                        Backends
                        IR
                        gtest
                        gtest_main)
add_test(IROptTest ${GLOW_BINARY_DIR}/tests/IROptTest)

add_executable(basicIRTest
                 IRTest.cpp)
target_link_libraries(basicIRTest
                      PRIVATE
                        Backends
                        IR
                        gtest
                        gtest_main)
add_test(basicIRTest ${GLOW_BINARY_DIR}/tests/basicIRTest)

add_executable(interpreterTest
                 InterpreterTest.cpp)
target_link_libraries(interpreterTest
                      PRIVATE
                        Backends
                        IR
                        ExecutionEngine
                        gtest
                        gtest_main)
add_test(interpreterTest ${GLOW_BINARY_DIR}/tests/interpreterTest)

add_executable(graphTest
                 GraphTest.cpp)
target_link_libraries(graphTest
                      PRIVATE
                        Graph
                        IR
                        Backends
                        gtest
                        gtest_main)
add_test(graphTest ${GLOW_BINARY_DIR}/tests/graphTest)

add_executable(graphGradTest
                 graphAutoGrad.cpp)
target_link_libraries(graphGradTest
                      PRIVATE
                        Graph
                        IR
                        Backends
                        ExecutionEngine
                        gtest
                        gtest_main)
add_test(graphGradTest ${GLOW_BINARY_DIR}/tests/graphGradTest)


add_executable(graphOptzTest
                 GraphOpt.cpp)
target_link_libraries(graphOptzTest
                      PRIVATE
                        Graph
                        IR
                        Backends
                        gtest
                        gtest_main)
add_test(graphOptzTest ${GLOW_BINARY_DIR}/tests/graphOptzTest)

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

add_executable(memoryAllocatorTest
                MemoryAllocatorTest.cpp)
target_link_libraries(memoryAllocatorTest
                      PRIVATE
                        Backends
                        IR
                        gtest
                        gtest_main)
add_test(memoryAllocatorTest ${GLOW_BINARY_DIR}/tests/memoryAllocatorTest)
