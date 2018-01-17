add_library(testMain
            testMain.cpp)
target_link_libraries(testMain
                      PRIVATE
                        gtest)

add_executable(tensorsTest
               tensorsTest.cpp)
target_link_libraries(tensorsTest
                      PRIVATE
                        Base
                        Support
                        gtest
                        testMain)
add_test(tensorsTest ${GLOW_BINARY_DIR}/tests/tensorsTest)

add_executable(gradCheckTest
               gradCheckTest.cpp)
target_link_libraries(gradCheckTest
                      PRIVATE
                        Backends
                        ExecutionEngine
                        IR
                        gtest
                        testMain)
add_test(gradCheckTest ${GLOW_BINARY_DIR}/tests/gradCheckTest)

add_executable(IROptTest
               IROptTest.cpp)
target_link_libraries(IROptTest
                      PRIVATE
                        Backends
                        IR
                        gtest
                        testMain)
add_test(IROptTest ${GLOW_BINARY_DIR}/tests/IROptTest)

add_executable(basicIRTest
               basicIRTest.cpp)
target_link_libraries(basicIRTest
                      PRIVATE
                        Backends
                        IR
                        gtest
                        testMain)
add_test(basicIRTest ${GLOW_BINARY_DIR}/tests/basicIRTest)

add_executable(interpreterTest
               InterpreterTest.cpp)
target_link_libraries(interpreterTest
                      PRIVATE
                        Backends
                        IR
                        ExecutionEngine
                        gtest
                        testMain)
add_test(interpreterTest ${GLOW_BINARY_DIR}/tests/interpreterTest)

add_executable(operatorTest
               OperatorTest.cpp)
target_link_libraries(operatorTest
                      PRIVATE
                        Backends
                        IR
                        ExecutionEngine
                        gtest
                        testMain)
add_test(operatorTest ${GLOW_BINARY_DIR}/tests/operatorTest)


add_executable(graphTest
               graphTest.cpp)
target_link_libraries(graphTest
                      PRIVATE
                        Graph
                        IR
                        Backends
                        gtest
                        testMain)
add_test(graphTest ${GLOW_BINARY_DIR}/tests/graphTest)

add_executable(graphGradTest
               graphGradTest.cpp)
target_link_libraries(graphGradTest
                      PRIVATE
                        Graph
                        IR
                        Backends
                        ExecutionEngine
                        gtest
                        testMain)
add_test(graphGradTest ${GLOW_BINARY_DIR}/tests/graphGradTest)


add_executable(graphOptzTest
               graphOptzTest.cpp)
target_link_libraries(graphOptzTest
                      PRIVATE
                        Graph
                        IR
                        Backends
                        gtest
                        testMain)
add_test(graphOptzTest ${GLOW_BINARY_DIR}/tests/graphOptzTest)

if(OpenCL_FOUND)
add_executable(OCLTest
               BackendTestUtils.cpp
               OCLTest.cpp)
target_link_libraries(OCLTest
                      PRIVATE
                        Backends
                        IR
                        ExecutionEngine
                        gtest
                        testMain)
add_test(OCLTest ${GLOW_BINARY_DIR}/tests/OCLTest)
endif()

add_executable(memoryAllocatorTest
               memoryAllocatorTest.cpp)
target_link_libraries(memoryAllocatorTest
                      PRIVATE
                        Backends
                        IR
                        gtest
                        testMain)
add_test(memoryAllocatorTest ${GLOW_BINARY_DIR}/tests/memoryAllocatorTest)

