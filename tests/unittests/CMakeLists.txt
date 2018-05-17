add_library(testMain
            testMain.cpp)
target_link_libraries(testMain
                      PRIVATE
                        LLVMSupport
                        gtest)

add_executable(tensorsTest
               tensorsTest.cpp)
target_link_libraries(tensorsTest
                      PRIVATE
                        Base
                        gtest
                        testMain)
add_test(tensorsTest ${GLOW_BINARY_DIR}/tests/tensorsTest)

add_executable(gradCheckTest
               gradCheckTest.cpp)
target_link_libraries(gradCheckTest
                      PRIVATE
                        Base
                        ExecutionEngine
                        Graph
                        IR
                        gtest
                        testMain)
add_test(gradCheckTest ${GLOW_BINARY_DIR}/tests/gradCheckTest)

add_executable(IROptTest
               IROptTest.cpp)
target_link_libraries(IROptTest
                      PRIVATE
                        Graph
                        IR
                        Optimizer
                        gtest
                        testMain)
add_test(IROptTest ${GLOW_BINARY_DIR}/tests/IROptTest)

add_executable(basicIRTest
               basicIRTest.cpp)
target_link_libraries(basicIRTest
                      PRIVATE
                        Graph
                        IR
                        gtest
                        testMain)
add_test(basicIRTest ${GLOW_BINARY_DIR}/tests/basicIRTest)

add_executable(backendTest
               BackendTest.cpp)
target_link_libraries(backendTest
                      PRIVATE
                        Graph
                        IR
                        ExecutionEngine
                        gtest
                        testMain)
add_test(backendTest ${GLOW_BINARY_DIR}/tests/backendTest)

add_executable(MLTest
	       MLTest.cpp)
target_link_libraries(MLTest
                      PRIVATE
                        Graph
                        IR
                        ExecutionEngine
                        gtest
                        testMain)
add_test(MLTest ${GLOW_BINARY_DIR}/tests/MLTest)

add_executable(operatorTest
               OperatorTest.cpp)
target_link_libraries(operatorTest
                      PRIVATE
                        Graph
                        IR
                        ExecutionEngine
                        gtest
                        testMain)
add_test(operatorTest ${GLOW_BINARY_DIR}/tests/operatorTest)


add_executable(graphTest
               graphTest.cpp)
target_link_libraries(graphTest
                      PRIVATE
                        ExecutionEngine
                        Graph
                        IR
                        gtest
                        testMain)
add_test(graphTest ${GLOW_BINARY_DIR}/tests/graphTest)

add_executable(graphGradTest
               graphGradTest.cpp)
target_link_libraries(graphGradTest
                      PRIVATE
                        Graph
                        IR
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
                        Optimizer
                        gtest
                        testMain)
add_test(graphOptzTest ${GLOW_BINARY_DIR}/tests/graphOptzTest)

add_executable(quantizationTest
               quantizationTest.cpp)
target_link_libraries(quantizationTest
                      PRIVATE
                        ExecutionEngine
                        Graph
                        Quantization
                        gtest
                        testMain)
add_test(quantizationTest ${GLOW_BINARY_DIR}/tests/quantizationTest)

add_executable(UtilsTest
               UtilsTest.cpp)
target_link_libraries(UtilsTest
                      PRIVATE
                        Support
                        gtest
                        testMain)
add_test(UtilsTest ${GLOW_BINARY_DIR}/tests/UtilsTest)

if(GLOW_WITH_OPENCL)
add_executable(OCLTest
               BackendTestUtils.cpp
               OCLTest.cpp)
target_link_libraries(OCLTest
                      PRIVATE
                        Graph
                        IR
                        ExecutionEngine
                        gtest
                        testMain)
add_test(OCLTest ${GLOW_BINARY_DIR}/tests/OCLTest)
LIST(APPEND UNOPT_TESTS ./tests/OCLTest -optimize-ir=false &&)
endif()

if(GLOW_WITH_CPU)
add_executable(JITTest
               BackendTestUtils.cpp
               JITTest.cpp)
target_link_libraries(JITTest
                      PRIVATE
                        Graph
                        IR
                        ExecutionEngine
                        Support
                        gtest
                        testMain)
add_test(JITTest ${GLOW_BINARY_DIR}/tests/JITTest)
LIST(APPEND UNOPT_TESTS ./tests/JITTest -optimize-ir=false &&)

add_executable(GemmTest
               GemmTest.cpp)
target_link_libraries(GemmTest
                      PRIVATE
                        Graph
                        IR
                        ExecutionEngine
                        Support
                        gtest
                        testMain)
add_test(GemmTest ${GLOW_BINARY_DIR}/tests/GemmTest)
LIST(APPEND UNOPT_TESTS ./tests/GemmTest -optimize-ir=false &&)
endif()

add_executable(memoryAllocatorTest
               memoryAllocatorTest.cpp)
target_link_libraries(memoryAllocatorTest
                      PRIVATE
                        CodeGen
                        gtest
                        testMain)
add_test(memoryAllocatorTest ${GLOW_BINARY_DIR}/tests/memoryAllocatorTest)


LIST(APPEND UNOPT_TESTS
       ./tests/backendTest -optimize-ir=false &&
       ./tests/MLTest -optimize-ir=false &&
       ./tests/operatorTest -optimize-ir=false &&
       ./tests/quantizationTest -optimize-ir=false &&
       ./tests/gradCheckTest -optimize-ir=false &&
       ./tests/graphTest -optimize-ir=false &&
       ./tests/graphGradTest -optimize-ir=false)

add_custom_target(test_unopt ${UNOPT_TESTS}
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
