add_library(TestMain
            TestMain.cpp)
target_link_libraries(TestMain
                      PRIVATE
                        LLVMSupport
                        gtest)

add_library(BackendTestUtils
            BackendTestUtils.cpp)
target_link_libraries(BackendTestUtils
                      PRIVATE
                        Base
                        Converter
                        ExecutionEngine
                        Graph
                        GraphOptimizer
                        Quantization
                        QuantizationBase
                        LLVMSupport
                        gtest)

# Test executables, sorted alphabetically.

add_executable(BackendCorrectnessTest
               BackendCorrectnessTest.cpp)
target_link_libraries(BackendCorrectnessTest
                      PRIVATE
                        Backends
                        BackendTestUtils
                        Graph
                        IR
                        ExecutionEngine
                        Quantization
                        Support
                        gtest
                        TestMain)
add_glow_test(BackendCorrectnessTest
              ${GLOW_BINARY_DIR}/tests/BackendCorrectnessTest --gtest_output=xml:BackendCorrectnessTest.xml)
LIST(APPEND UNOPT_TESTS ./tests/BackendCorrectnessTest -optimize-ir=false &&)

add_executable(BackendTest
               BackendTest.cpp)
target_link_libraries(BackendTest
                      PRIVATE
                        Backends
                        Graph
                        IR
                        ExecutionEngine
                        gtest
                        TestMain)
add_glow_test(BackendTest ${GLOW_BINARY_DIR}/tests/BackendTest --gtest_output=xml:BackendTest.xml)
LIST(APPEND UNOPT_TESTS ./tests/BackendTest -optimize-ir=false &&)

add_executable(BasicIRTest
               BasicIRTest.cpp)
target_link_libraries(BasicIRTest
                      PRIVATE
                        Graph
                        IR
                        gtest
                        TestMain)
add_glow_test(BasicIRTest ${GLOW_BINARY_DIR}/tests/BasicIRTest --gtest_output=xml:BasicIRTest.xml)

add_executable(Caffe2ImporterTest
               Caffe2ImporterTest.cpp)
target_link_libraries(Caffe2ImporterTest
                      PRIVATE
                        Graph
                        Importer
                        ExecutionEngine
                        gtest
                        TestMain)
add_glow_test(NAME Caffe2ImporterTest
              COMMAND ${GLOW_BINARY_DIR}/tests/Caffe2ImporterTest --gtest_output=xml:Caffe2ImporterTest.xml
              WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(DeviceManagerTest
               DeviceManagerTest.cpp)
target_link_libraries(DeviceManagerTest
                      PRIVATE
                        Backend
                        Backends
                        Graph
                        IR
                        ExecutionEngine
                        GraphOptimizer
                        gtest
                        TestMain)
add_glow_test(DeviceManagerTest ${GLOW_BINARY_DIR}/tests/DeviceManagerTest --gtest_output=xml:DeviceManagerTest.xml)

add_executable(ThreadPoolExecutorTest
        ThreadPoolExecutorTest.cpp)
target_link_libraries(ThreadPoolExecutorTest
        PRIVATE
        Backends
        Executor
        Graph
        gtest
        TestMain)
add_glow_test(ThreadPoolExecutorTest ${GLOW_BINARY_DIR}/tests/ThreadPoolExecutorTest --gtest_output=xml:ThreadPoolExecutorTest.xml)

add_executable(Float16Test
               Float16Test.cpp)
target_link_libraries(Float16Test
                      PRIVATE
                        Support
                        gtest
                        TestMain)
add_glow_test(Float16Test ${GLOW_BINARY_DIR}/tests/Float16Test --gtest_output=xml:Float16Test.xml)

if(GLOW_WITH_CPU AND NOT MSVC)
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
                          TestMain)
  add_glow_test(GemmTest ${GLOW_BINARY_DIR}/tests/GemmTest --gtest_output=xml:GemmLTest.xml)
  LIST(APPEND UNOPT_TESTS ./tests/GemmTest -optimize-ir=false &&)
endif()

add_executable(GlowOnnxifiManagerTest
               GlowOnnxifiManagerTest.cpp)
target_link_libraries(GlowOnnxifiManagerTest
                      PRIVATE
                        onnxifi-glow-lib
                        gtest
                        TestMain)
add_glow_test(NAME GlowOnnxifiManagerTest
              COMMAND ${GLOW_BINARY_DIR}/tests/GlowOnnxifiManagerTest --gtest_output=xml:GlowOnnxifiManagerTest.xml
              WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(GradCheckTest
               GradCheckTest.cpp)
target_link_libraries(GradCheckTest
                      PRIVATE
                        Base
                        ExecutionEngine
                        Graph
                        IR
                        gtest
                        TestMain)
add_glow_test(GradCheckTest ${GLOW_BINARY_DIR}/tests/GradCheckTest --gtest_output=xml:GradCheckTest.xml)
LIST(APPEND UNOPT_TESTS ./tests/GradCheckTest -optimize-ir=false &&)

add_executable(GraphGradTest
               GraphGradTest.cpp)
target_link_libraries(GraphGradTest
                      PRIVATE
                        Graph
                        IR
                        ExecutionEngine
                        gtest
                        TestMain)
add_glow_test(GraphGradTest ${GLOW_BINARY_DIR}/tests/GraphGradTest --gtest_output=xml:GraphGradTest.xml)
LIST(APPEND UNOPT_TESTS ./tests/GraphGradTest -optimize-ir=false &&)

add_executable(GraphOptzTest
               GraphOptzTest.cpp)
target_link_libraries(GraphOptzTest
                      PRIVATE
                        ExecutionEngine
                        Graph
                        IR
                        GraphOptimizer
                        gtest
                        TestMain)
add_glow_test(GraphOptzTest ${GLOW_BINARY_DIR}/tests/GraphOptzTest --gtest_output=xml:GraphOptzTest.xml)

add_executable(GraphSchedulerTest
               GraphSchedulerTest.cpp)
target_link_libraries(GraphSchedulerTest
                      PRIVATE
                        Graph
                        IR
                        gtest
                        TestMain)
add_glow_test(GraphSchedulerTest ${GLOW_BINARY_DIR}/tests/GraphSchedulerTest --gtest_output=xml:GraphSchedulerTest.xml)

add_executable(GraphTest
               GraphTest.cpp)
target_link_libraries(GraphTest
                      PRIVATE
                        Backend
                        Backends
                        ExecutionEngine
                        Graph
                        IR
                        GraphOptimizer
                        gtest
                        TestMain)
add_glow_test(GraphTest ${GLOW_BINARY_DIR}/tests/GraphTest --gtest_output=xml:GraphTest.xml)
LIST(APPEND UNOPT_TESTS ./tests/GraphTest -optimize-ir=false &&)

if(GLOW_WITH_HABANA)
  add_executable(HabanaBackendTest
                 HabanaBackendTest.cpp)
  target_link_libraries(HabanaBackendTest
                        PRIVATE
                          Backends
                          ExecutionEngine
                          Graph
                          GraphOptimizer
                          gtest
                          TestMain)
  target_include_directories(HabanaBackendTest PRIVATE "${SYNAPSE_INCLUDE_DIR}")
  add_glow_test(HabanaBackendTest ${GLOW_BINARY_DIR}/tests/HabanaBackendTest --gtest_output=xml:HabanaBackendTest.xml)

  add_executable(HabanaTest
          HabanaTest.cpp)
  target_link_libraries(HabanaTest
          PRIVATE
          Habana
          Support
          gtest
          TestMain)
  add_glow_test(HabanaTest ${GLOW_BINARY_DIR}/tests/HabanaTest --gtest_output=xml:HabanaTest.xml)
endif()

if(GLOW_WITH_CPU)
  add_executable(HostManagerTest
          HostManagerTest.cpp)
  target_link_libraries(HostManagerTest
          PRIVATE
          Backends
          HostManager
          ExecutionEngine
          Executor
          Graph
          IR
          Partitioner
          Provisioner
          gtest
          TestMain)

  add_glow_test(HostManagerTest ${GLOW_BINARY_DIR}/tests/HostManagerTest --gtest_output=xml:ProvisionerTest.xml)

  add_executable(HyphenTest
                 HyphenTest.cpp)
  target_link_libraries(HyphenTest
                        PRIVATE
                          BackendTestUtils
                          Graph
                          IR
                          ExecutionEngine
                          Quantization
                          Support
                          gtest
                          TestMain)
  add_glow_test(HyphenTest ${GLOW_BINARY_DIR}/tests/HyphenTest --gtest_output=xml:HyphenTest.xml)
  list(APPEND UNOPT_TESTS ./tests/HyphenTest -optimize-ir=false &&)
endif()

add_executable(IROptTest
               IROptTest.cpp)
target_link_libraries(IROptTest
                      PRIVATE
                        Backend
                        Graph
                        IR
                        IROptimizer
                        gtest
                        TestMain)
add_glow_test(IROptTest ${GLOW_BINARY_DIR}/tests/IROptTest --gtest_output=xml:IROptTest.xml)

if(PNG_FOUND)
  add_executable(ImageTest
          ImageTest.cpp)
  target_link_libraries(ImageTest
          PRIVATE
          Base
          gtest
          TestMain)
  add_glow_test(ImageTest ${GLOW_BINARY_DIR}/tests/ImageTest --gtest_output=xml:ImageTest.xml
          WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

if(GLOW_WITH_CPU)
  add_executable(LLVMIRGenTest
                 LLVMIRGenTest.cpp)
  target_link_libraries(LLVMIRGenTest
                        PRIVATE
                          Backend
                          CPUBackend
                          IR
                          Support
                          gtest
                          TestMain)
  add_glow_test(LLVMIRGenTest ${GLOW_BINARY_DIR}/tests/LLVMIRGenTest --gtest_output=xml:LLVMIRGenTest.xml)
endif()

add_executable(MLTest
        MLTest.cpp)
target_link_libraries(MLTest
        PRIVATE
        Graph
        IR
        ExecutionEngine
        Quantization
        gtest
        TestMain)
add_glow_test(MLTest ${GLOW_BINARY_DIR}/tests/MLTest --gtest_output=xml:MLTest.xml)
LIST(APPEND UNOPT_TESTS ./tests/MLTest -optimize-ir=false &&)

add_executable(MemoryAllocatorTest
               MemoryAllocatorTest.cpp)
target_link_libraries(MemoryAllocatorTest
                      PRIVATE
                        CodeGen
                        gtest
                        Support
                        TestMain)
add_glow_test(MemoryAllocatorTest ${GLOW_BINARY_DIR}/tests/MemoryAllocatorTest --gtest_output=xml:MemoryAllocatorTest.xml)



if(GLOW_WITH_OPENCL)
  add_executable(OCLTest
                 OCLTest.cpp)
  target_link_libraries(OCLTest
                        PRIVATE
                          Backends
                          BackendTestUtils
                          ExecutionEngine
                          Graph
                          IR
                          gtest
                          TestMain)
  add_glow_test(OCLTest ${GLOW_BINARY_DIR}/tests/OCLTest --gtest_output=xml:OCLTest.xml)
  LIST(APPEND UNOPT_TESTS ./tests/OCLTest -optimize-ir=false &&)
endif()

add_executable(OnnxImporterTest
               OnnxImporterTest.cpp)
target_link_libraries(OnnxImporterTest
                      PRIVATE
                        Backend
                        Graph
                        Importer
                        ExecutionEngine
                        gtest
                        TestMain)
add_glow_test(NAME OnnxImporterTest
              COMMAND ${GLOW_BINARY_DIR}/tests/OnnxImporterTest --gtest_output=xml:OnnxImporterTest.xml
              WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(OperatorGradTest
        OperatorGradTest.cpp)
target_link_libraries(OperatorGradTest
        PRIVATE
        Base
        ExecutionEngine
        Graph
        gtest
        TestMain)
add_glow_test(OperatorGradTest ${GLOW_BINARY_DIR}/tests/OperatorGradTest --gtest_output=xml:OperatorGradTest.xml)
LIST(APPEND UNOPT_TESTS ./tests/OperatorGradTest -optimize-ir=false &&)

add_executable(OperatorTest
        OperatorTest.cpp)
target_link_libraries(OperatorTest
        PRIVATE
        BackendTestUtils
        Graph
        IR
        ExecutionEngine
        Quantization
        GraphOptimizer
        gtest
        TestMain)
add_glow_test(OperatorTest ${GLOW_BINARY_DIR}/tests/OperatorTest --gtest_output=xml:OperatorTest.xml)
LIST(APPEND UNOPT_TESTS ./tests/OperatorTest -optimize-ir=false &&)

add_executable(PartitionerTest
               PartitionerTest.cpp)
target_link_libraries(PartitionerTest
                      PRIVATE
                        Backend
                        Backends
                        ExecutionEngine
                        Graph
                        IR
                        Partitioner
                        gtest
                        TestMain)
add_glow_test(PartitionerTest ${GLOW_BINARY_DIR}/tests/PartitionerTest --gtest_output=xml:PartitionerTest.xml)

add_executable(RecommendationSystemTest
               RecommendationSystemTest.cpp)
target_link_libraries(RecommendationSystemTest
                      PRIVATE
                        BackendTestUtils
                        Converter
                        Graph
                        IR
                        ExecutionEngine
                        Quantization
                        Partitioner
                        gtest
                        TestMain)
add_glow_test(EXPENSIVE RecommendationSystemTest ${GLOW_BINARY_DIR}/tests/RecommendationSystemTest --gtest_output=xml:RecommendationSystemTest.xml)

if(GLOW_WITH_CPU)
  add_executable(ProvisionerTest
                 ProvisionerTest.cpp)
  target_link_libraries(ProvisionerTest
                        PRIVATE
                          Backends
                          ExecutionEngine
                          Graph
                          IR
                          Provisioner
                          GraphOptimizer
                          gtest
                          TestMain)
  add_glow_test(ProvisionerTest ${GLOW_BINARY_DIR}/tests/ProvisionerTest --gtest_output=xml:ProvisionerTest.xml)

endif()

add_executable(QuantizationTest
               QuantizationTest.cpp)
target_link_libraries(QuantizationTest
                      PRIVATE
                        Backend
                        Backends
                        ExecutionEngine
                        Graph
                        GraphOptimizer
                        Quantization
                        gtest
                        TestMain)
add_glow_test(QuantizationTest ${GLOW_BINARY_DIR}/tests/QuantizationTest --gtest_output=xml:QuantizationTest.xml)
LIST(APPEND UNOPT_TESTS ./tests/QuantizationTest -optimize-ir=false &&)

add_executable(TensorsTest
               TensorsTest.cpp)
target_link_libraries(TensorsTest
                      PRIVATE
                        Backend
                        Base
                        gtest
                        TestMain)
add_glow_test(TensorsTest ${GLOW_BINARY_DIR}/tests/TensorsTest --gtest_output=xml:TensorsTest.xml)

add_executable(TensorPoolTest
               TensorPoolTest.cpp)
target_link_libraries(TensorPoolTest
                      PRIVATE
                        Graph
                        TensorPool
                        gtest
                        TestMain)
add_glow_test(TensorPoolTest ${GLOW_BINARY_DIR}/tests/TensorPoolTest --gtest_output=xml:TensorPoolTest.xml)

add_executable(ThreadPoolTest
               ThreadPoolTest.cpp)
target_link_libraries(ThreadPoolTest
                      PRIVATE
                        Support
                        gtest
                        TestMain)
add_glow_test(ThreadPoolTest ${GLOW_BINARY_DIR}/tests/ThreadPoolTest --gtest_output=xml:ThreadPoolTest.xml)

add_executable(TraceEventsTest
               TraceEventsTest.cpp)
target_link_libraries(TraceEventsTest
                      PRIVATE
                        Backend
                        Graph
                        IR
                        ExecutionEngine
                        GraphOptimizer
                        gtest
                        TestMain)
add_glow_test(TraceEventsTest ${GLOW_BINARY_DIR}/tests/TraceEventsTest --gtest_output=xml:TraceEventsTest.xml)
LIST(APPEND UNOPT_TESTS ./tests/TraceEventsTest -optimize-ir=false &&)

add_executable(TypeAToTypeBFunctionConverterTest
               TypeAToTypeBFunctionConverterTest.cpp)
target_link_libraries(TypeAToTypeBFunctionConverterTest
                      PRIVATE
                        Converter
                        ExecutionEngine
                        Graph
                        GraphOptimizer
                        gtest
                        TestMain)
add_glow_test(TypeAToTypeBFunctionConverterTest
              ${GLOW_BINARY_DIR}/tests/TypeAToTypeBFunctionConverterTest --gtest_output=xml:TypeAToTypeBFunctionConverterTest.xml)

add_executable(UtilsTest
               StrCheck.cpp
               SupportTest.cpp
               TaggedListTest.cpp
               UtilsTest.cpp)
target_link_libraries(UtilsTest
                      PRIVATE
                        Base
                        Support
                        Testing
                        gtest
                        TestMain)
add_glow_test(NAME UtilsTest
              COMMAND ${GLOW_BINARY_DIR}/tests/UtilsTest --gtest_output=xml:UtilsTest.xml
              WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

LIST(APPEND UNOPT_TESTS true)
add_custom_target(test_unopt ${UNOPT_TESTS}
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
