add_library(testMain
            testMain.cpp)
target_link_libraries(testMain
                      PRIVATE
                        LLVMSupport
                        gtest)

add_executable(partitionTest
               partitionTest.cpp)
target_link_libraries(partitionTest
                      PRIVATE
                        ExecutionEngine
                        gtest
                        testMain)
add_glow_test(partitionTest ${GLOW_BINARY_DIR}/tests/partitionTest)

add_executable(tensorsTest
               tensorsTest.cpp)
target_link_libraries(tensorsTest
                      PRIVATE
                        Base
                        gtest
                        testMain)
add_glow_test(tensorsTest ${GLOW_BINARY_DIR}/tests/tensorsTest)

if(PNG_FOUND)
add_executable(imageTest
               imageTest.cpp)
target_link_libraries(imageTest
                      PRIVATE
                        Base
                        gtest
                        testMain)
add_glow_test(imageTest ${GLOW_BINARY_DIR}/tests/imageTest
              WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

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
add_glow_test(gradCheckTest ${GLOW_BINARY_DIR}/tests/gradCheckTest)

add_executable(IROptTest
               IROptTest.cpp)
target_link_libraries(IROptTest
                      PRIVATE
                        Graph
                        IR
                        Optimizer
                        gtest
                        testMain)
add_glow_test(IROptTest ${GLOW_BINARY_DIR}/tests/IROptTest)

add_executable(basicIRTest
               basicIRTest.cpp)
target_link_libraries(basicIRTest
                      PRIVATE
                        Graph
                        IR
                        gtest
                        testMain)
add_glow_test(basicIRTest ${GLOW_BINARY_DIR}/tests/basicIRTest)

add_executable(backendTest
               BackendTest.cpp)
target_link_libraries(backendTest
                      PRIVATE
                        Graph
                        IR
                        ExecutionEngine
                        gtest
                        testMain)
add_glow_test(backendTest ${GLOW_BINARY_DIR}/tests/backendTest)

add_executable(MLTest
               MLTest.cpp)
target_link_libraries(MLTest
                      PRIVATE
                        Graph
                        IR
                        ExecutionEngine
                        Quantization
                        gtest
                        testMain)
add_glow_test(MLTest ${GLOW_BINARY_DIR}/tests/MLTest)

add_executable(operatorTest
               OperatorTest.cpp)
target_link_libraries(operatorTest
                      PRIVATE
                        Graph
                        IR
                        ExecutionEngine
                        Quantization
                        gtest
                        testMain)
add_glow_test(operatorTest ${GLOW_BINARY_DIR}/tests/operatorTest)

add_executable(graphTest
               graphTest.cpp)
target_link_libraries(graphTest
                      PRIVATE
                        ExecutionEngine
                        Graph
                        IR
                        gtest
                        testMain)
add_glow_test(graphTest ${GLOW_BINARY_DIR}/tests/graphTest)

add_executable(graphGradTest
               graphGradTest.cpp)
target_link_libraries(graphGradTest
                      PRIVATE
                        Graph
                        IR
                        ExecutionEngine
                        gtest
                        testMain)
add_glow_test(graphGradTest ${GLOW_BINARY_DIR}/tests/graphGradTest)

add_executable(graphOptzTest
               graphOptzTest.cpp)
target_link_libraries(graphOptzTest
                      PRIVATE
                        Graph
                        IR
                        Optimizer
                        gtest
                        testMain)
add_glow_test(graphOptzTest ${GLOW_BINARY_DIR}/tests/graphOptzTest)

add_executable(graphSchedulerTest
               graphSchedulerTest.cpp)
target_link_libraries(graphSchedulerTest
                      PRIVATE
                        Graph
                        IR
                        gtest
                        testMain)
target_include_directories(graphSchedulerTest PRIVATE ${CMAKE_SOURCE_DIR}/lib/IR)
add_glow_test(graphSchedulerTest ${GLOW_BINARY_DIR}/tests/graphSchedulerTest)

add_executable(quantizationTest
               quantizationTest.cpp)
target_link_libraries(quantizationTest
                      PRIVATE
                        ExecutionEngine
                        Graph
                        Quantization
                        gtest
                        testMain)
add_glow_test(quantizationTest ${GLOW_BINARY_DIR}/tests/quantizationTest)

add_executable(float16Test
               float16Test.cpp)
target_link_libraries(float16Test
                      PRIVATE
                        Support
                        gtest
                        testMain)
add_glow_test(float16Test ${GLOW_BINARY_DIR}/tests/float16Test)

add_executable(typeAToTypeBFunctionConverterTest
               typeAToTypeBFunctionConverterTest.cpp)
target_link_libraries(typeAToTypeBFunctionConverterTest
                      PRIVATE
                        Converter
                        ExecutionEngine
                        Graph
                        gtest
                        testMain)
add_glow_test(typeAToTypeBFunctionConverterTest
              ${GLOW_BINARY_DIR}/tests/typeAToTypeBFunctionConverterTest)

add_executable(UtilsTest
               UtilsTest.cpp)
target_link_libraries(UtilsTest
                      PRIVATE
                        Base
                        Support
                        gtest
                        testMain)
add_glow_test(UtilsTest ${GLOW_BINARY_DIR}/tests/UtilsTest)

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
add_glow_test(OCLTest ${GLOW_BINARY_DIR}/tests/OCLTest)
LIST(APPEND UNOPT_TESTS ./tests/OCLTest -optimize-ir=false &&)
endif()

add_executable(convCorrectnessTest
               convTest.cpp)
target_link_libraries(convCorrectnessTest
                      PRIVATE
                        Graph
                        IR
                        ExecutionEngine
                        Support
                        gtest
                        testMain)

add_glow_test(convCorrectnessTest
              ${GLOW_BINARY_DIR}/tests/convCorrectnessTest)

LIST(APPEND UNOPT_TESTS ./tests/convCorrectnessTest -optimize-ir=false &&)

add_executable(BackendCorrectnessTest
               BackendTestUtils.cpp
               BackendCorrectnessTest.cpp)
target_link_libraries(BackendCorrectnessTest
                      PRIVATE
                        Graph
                        IR
                        ExecutionEngine
                        Support
                        Quantization
                        gtest
                        testMain)

add_glow_test(BackendCorrectnessTest
              ${GLOW_BINARY_DIR}/tests/BackendCorrectnessTest)

LIST(APPEND UNOPT_TESTS ./tests/BackendCorrectnessTest -optimize-ir=false &&)

if(GLOW_WITH_CPU)
target_sources(BackendCorrectnessTest PUBLIC HyphenTest.cpp)

if (NOT MSVC)
add_executable(GemmTest
               GemmTest.cpp)
target_link_libraries(GemmTest
                      PRIVATE
                        CPURuntimeNative
                        Graph
                        IR
                        ExecutionEngine
                        Support
                        gtest
                        testMain)
add_glow_test(GemmTest ${GLOW_BINARY_DIR}/tests/GemmTest)
LIST(APPEND UNOPT_TESTS ./tests/GemmTest -optimize-ir=false &&)
endif()

add_executable(LLVMIRGenTest
               LLVMIRGenTest.cpp)
target_link_libraries(LLVMIRGenTest
                      PRIVATE
                        CPUBackend
                        IR
                        Support
                        gtest
                        testMain)
target_include_directories(LLVMIRGenTest PUBLIC ${CMAKE_SOURCE_DIR}/lib/Backends/CPU)
add_glow_test(LLVMIRGenTest ${GLOW_BINARY_DIR}/tests/LLVMIRGenTest)

endif()

add_executable(memoryAllocatorTest
               memoryAllocatorTest.cpp)
target_link_libraries(memoryAllocatorTest
                      PRIVATE
                        CodeGen
                        gtest
                        Support
                        testMain)
add_glow_test(memoryAllocatorTest ${GLOW_BINARY_DIR}/tests/memoryAllocatorTest)

add_executable(caffe2ImporterTest
               caffe2ImporterTest.cpp)
target_link_libraries(caffe2ImporterTest
                      PRIVATE
                        Graph
                        Importer
                        ExecutionEngine
                        gtest
                        testMain)
add_glow_test(NAME caffe2ImporterTest
              COMMAND ${GLOW_BINARY_DIR}/tests/caffe2ImporterTest
              WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(onnxImporterTest
               onnxImporterTest.cpp)
target_link_libraries(onnxImporterTest
                      PRIVATE
                        Graph
                        Importer
                        ExecutionEngine
                        gtest
                        testMain)
add_glow_test(NAME onnxImporterTest
              COMMAND ${GLOW_BINARY_DIR}/tests/onnxImporterTest
              WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(onnxifiThreadPoolTest
               onnxifiThreadPoolTest.cpp)
target_link_libraries(onnxifiThreadPoolTest
                      PRIVATE
                        onnxifi-glow-thread-pool
                        gtest
                        testMain)
target_include_directories(onnxifiThreadPoolTest PRIVATE ${CMAKE_SOURCE_DIR}/lib/Onnxifi)
add_glow_test(NAME onnxifiThreadPoolTest
              COMMAND ${GLOW_BINARY_DIR}/tests/onnxifiThreadPoolTest
              WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

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
