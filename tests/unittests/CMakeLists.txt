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
                        Backends
                        ExecutionEngine
                        Graph
                        IR
                        Optimizer
                        gtest
                        testMain)
add_glow_test(partitionTest ${GLOW_BINARY_DIR}/tests/partitionTest --gtest_output=xml:partitionTest.xml)

add_executable(tensorsTest
               tensorsTest.cpp)
target_link_libraries(tensorsTest
                      PRIVATE
                        Base
                        gtest
                        testMain)
add_glow_test(tensorsTest ${GLOW_BINARY_DIR}/tests/tensorsTest --gtest_output=xml:tensorsTest.xml)

if(PNG_FOUND)
add_executable(imageTest
               imageTest.cpp)
target_link_libraries(imageTest
                      PRIVATE
                        Base
                        gtest
                        testMain)
add_glow_test(imageTest ${GLOW_BINARY_DIR}/tests/imageTest --gtest_output=xml:imageTest.xml
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
add_glow_test(gradCheckTest ${GLOW_BINARY_DIR}/tests/gradCheckTest --gtest_output=xml:gradCheckTest.xml)

add_executable(IROptTest
               IROptTest.cpp)
target_link_libraries(IROptTest
                      PRIVATE
                        Graph
                        IR
                        Optimizer
                        gtest
                        testMain)
add_glow_test(IROptTest ${GLOW_BINARY_DIR}/tests/IROptTest --gtest_output=xml:IROptTest.xml)

add_executable(basicIRTest
               basicIRTest.cpp)
target_link_libraries(basicIRTest
                      PRIVATE
                        Graph
                        IR
                        gtest
                        testMain)
add_glow_test(basicIRTest ${GLOW_BINARY_DIR}/tests/basicIRTest --gtest_output=xml:basicIRTest.xml) 

add_executable(backendTest
               BackendTest.cpp)
target_link_libraries(backendTest
                      PRIVATE
                        Backends
                        Graph
                        IR
                        ExecutionEngine
                        Optimizer
                        gtest
                        testMain)
add_glow_test(backendTest ${GLOW_BINARY_DIR}/tests/backendTest --gtest_output=xml:backendTest.xml)

add_executable(MLTest
               MLTest.cpp)
target_link_libraries(MLTest
                      PRIVATE
                        Graph
                        IR
                        ExecutionEngine
                        Optimizer
                        Quantization
                        gtest
                        testMain)
add_glow_test(MLTest ${GLOW_BINARY_DIR}/tests/MLTest --gtest_output=xml:MLTest.xml)

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
add_glow_test(operatorTest ${GLOW_BINARY_DIR}/tests/operatorTest --gtest_output=xml:operatorTest.xml)

add_executable(graphTest
               graphTest.cpp)
target_link_libraries(graphTest
                      PRIVATE
                        Backends
                        ExecutionEngine
                        Graph
                        IR
                        Optimizer
                        gtest
                        testMain)
add_glow_test(graphTest ${GLOW_BINARY_DIR}/tests/graphTest --gtest_output=xml:graphTest.xml)

add_executable(graphGradTest
               graphGradTest.cpp)
target_link_libraries(graphGradTest
                      PRIVATE
                        Graph
                        IR
                        ExecutionEngine
                        gtest
                        testMain)
add_glow_test(graphGradTest ${GLOW_BINARY_DIR}/tests/graphGradTest --gtest_output=xml:graphGradTest.xml)

add_executable(graphOptzTest
               graphOptzTest.cpp)
target_link_libraries(graphOptzTest
                      PRIVATE
                        Graph
                        IR
                        Optimizer
                        gtest
                        testMain)
add_glow_test(graphOptzTest ${GLOW_BINARY_DIR}/tests/graphOptzTest --gtest_output=xml:graphOptzTest.xml) 

add_executable(graphSchedulerTest
               graphSchedulerTest.cpp)
target_link_libraries(graphSchedulerTest
                      PRIVATE
                        Graph
                        IR
                        gtest
                        testMain)
target_include_directories(graphSchedulerTest PRIVATE ${CMAKE_SOURCE_DIR}/lib/IR)
add_glow_test(graphSchedulerTest ${GLOW_BINARY_DIR}/tests/graphSchedulerTest --gtest_output=xml:graphSchedulerTest.xml)

add_executable(quantizationTest
               quantizationTest.cpp)
target_link_libraries(quantizationTest
                      PRIVATE
                        Backends
                        ExecutionEngine
                        Graph
                        Optimizer
                        Quantization
                        gtest
                        testMain)
add_glow_test(quantizationTest ${GLOW_BINARY_DIR}/tests/quantizationTest --gtest_output=xml:quantizationTest.xml)

add_executable(float16Test
               float16Test.cpp)
target_link_libraries(float16Test
                      PRIVATE
                        Support
                        gtest
                        testMain)
add_glow_test(float16Test ${GLOW_BINARY_DIR}/tests/float16Test --gtest_output=xml:float16Test.xml)

add_executable(typeAToTypeBFunctionConverterTest
               typeAToTypeBFunctionConverterTest.cpp)
target_link_libraries(typeAToTypeBFunctionConverterTest
                      PRIVATE
                        Converter
                        ExecutionEngine
                        Graph
                        Optimizer
                        gtest
                        testMain)
add_glow_test(typeAToTypeBFunctionConverterTest
              ${GLOW_BINARY_DIR}/tests/typeAToTypeBFunctionConverterTest --gtest_output=xml:typeAToTypeBFunctionConverterTest.xml) 

add_executable(UtilsTest
               UtilsTest.cpp)
target_link_libraries(UtilsTest
                      PRIVATE
                        Base
                        Support
                        gtest
                        testMain)
add_glow_test(UtilsTest ${GLOW_BINARY_DIR}/tests/UtilsTest --gtest_output=xml:UtilsTest.xml)

if(GLOW_WITH_OPENCL)
add_executable(OCLTest
               BackendTestUtils.cpp
               OCLTest.cpp)
target_link_libraries(OCLTest
                      PRIVATE
                        ExecutionEngine
                        Graph
                        IR
                        Optimizer
                        gtest
                        testMain)
add_glow_test(OCLTest ${GLOW_BINARY_DIR}/tests/OCLTest --gtest_output=xml:OCLTest.xml)
LIST(APPEND UNOPT_TESTS ./tests/OCLTest -optimize-ir=false &&)
endif()

add_executable(convCorrectnessTest
               convTest.cpp)
target_link_libraries(convCorrectnessTest
                      PRIVATE
                        Graph
                        IR
                        ExecutionEngine
                        Optimizer
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
                        Backends
                        Graph
                        IR
                        ExecutionEngine
                        Optimizer
                        Quantization
                        Support
                        gtest
                        testMain)

add_glow_test(BackendCorrectnessTest
              ${GLOW_BINARY_DIR}/tests/BackendCorrectnessTest --gtest_output=xml:BackendCorrectnessTest.xml)

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
add_glow_test(GemmTest ${GLOW_BINARY_DIR}/tests/GemmTest --gtest_output=xml:GemmLTest.xml)
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
add_glow_test(LLVMIRGenTest ${GLOW_BINARY_DIR}/tests/LLVMIRGenTest --gtest_output=xml:LLVMIRGenTest.xml)

endif()

add_executable(memoryAllocatorTest
               memoryAllocatorTest.cpp)
target_link_libraries(memoryAllocatorTest
                      PRIVATE
                        CodeGen
                        gtest
                        Support
                        testMain)
add_glow_test(memoryAllocatorTest ${GLOW_BINARY_DIR}/tests/memoryAllocatorTest --gtest_output=xml:memoryAllocatorTest.xml)

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
              COMMAND ${GLOW_BINARY_DIR}/tests/caffe2ImporterTest --gtest_output=xml:Caffe2ImporterTest.xml
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
              COMMAND ${GLOW_BINARY_DIR}/tests/onnxImporterTest --gtest_output=xml:onnxImporterTest.xml
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
              COMMAND ${GLOW_BINARY_DIR}/tests/onnxifiThreadPoolTest --gtest_output=xml:onnxifiThreadPoolTest.xml
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
