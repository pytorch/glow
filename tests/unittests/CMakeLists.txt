add_library(TestMain
            TestMain.cpp)
target_link_libraries(TestMain
                      PRIVATE
                        LLVMSupport
                        gtest)

add_library(BackendTestUtils
            BackendTestUtils.cpp)
target_link_libraries(BackendTestUtils
                      PUBLIC
                        Backend
                        Backends
                        Base
                        Converter
                        ExecutionEngine
                        Graph
                        GraphOptimizer
                        Quantization
                        QuantizationBase
                        LLVMSupport
                        gtest)

# Repro binary
add_executable(Repro
               Repro.cpp)
target_link_libraries(Repro
                      PRIVATE
                        BackendTestUtils
                        HostManager
                        Importer
                        Exporter)

# Test executables, sorted alphabetically.

add_executable(BasicIRTest
               BasicIRTest.cpp)
target_link_libraries(BasicIRTest
                      PRIVATE
                        Backend
                        Backends
                        Graph
                        IR
                        gtest
                        TestMain)
add_glow_test(BasicIRTest
              ${GLOW_BINARY_DIR}/tests/BasicIRTest
                  --gtest_output=xml:BasicIRTest.xml)

if(GLOW_WITH_LLVMIRCODEGEN)
  add_executable(BundleSaverTest
                 BundleSaverTest.cpp)
  target_link_libraries(BundleSaverTest
                        PRIVATE
                          Backends
                          Graph
                          Support
                          gtest
                          TestMain)
  add_glow_test(BundleSaverDynamicAPITest
                ${GLOW_BINARY_DIR}/tests/BundleSaverTest
                    --gtest_output=xml:BundleSaverDynamicAPITest.xml
                    -bundle-api=dynamic)
  add_glow_test(BundleSaverStaticAPITest
                ${GLOW_BINARY_DIR}/tests/BundleSaverTest
                    --gtest_output=xml:BundleSaverStaticAPITest.xml
                    -bundle-api=static)
endif()

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
              COMMAND ${GLOW_BINARY_DIR}/tests/Caffe2ImporterTest
                  --gtest_output=xml:Caffe2ImporterTest.xml
              WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(ThreadPoolExecutorTest
        ThreadPoolExecutorTest.cpp)
target_link_libraries(ThreadPoolExecutorTest
        PRIVATE
        Backends
        Executor
        Graph
        gtest
        TestMain)
add_glow_test(ThreadPoolExecutorTest
              ${GLOW_BINARY_DIR}/tests/ThreadPoolExecutorTest
                  --gtest_output=xml:ThreadPoolExecutorTest.xml)

add_executable(Float16Test
               Float16Test.cpp)
target_link_libraries(Float16Test
                      PRIVATE
                        Support
                        gtest
                        TestMain)
add_glow_test(Float16Test
              ${GLOW_BINARY_DIR}/tests/Float16Test
                  --gtest_output=xml:Float16Test.xml)

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
  add_glow_test(GemmTest
                ${GLOW_BINARY_DIR}/tests/GemmTest
                    --gtest_output=xml:GemmLTest.xml)
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
              COMMAND ${GLOW_BINARY_DIR}/tests/GlowOnnxifiManagerTest
                  --gtest_output=xml:GlowOnnxifiManagerTest.xml
              WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(GraphGradTest
               GraphGradTest.cpp)
target_link_libraries(GraphGradTest
                      PRIVATE
                        Graph
                        IR
                        ExecutionEngine
                        gtest
                        TestMain)
add_glow_test(GraphGradTest
              ${GLOW_BINARY_DIR}/tests/GraphGradTest
                  --gtest_output=xml:GraphGradTest.xml)
LIST(APPEND UNOPT_TESTS ./tests/GraphGradTest -optimize-ir=false &&)

add_executable(GraphOptzTest
               GraphOptzTest.cpp)
target_link_libraries(GraphOptzTest
                      PRIVATE
                        Backends
                        BackendTestUtils
                        ExecutionEngine
                        Graph
                        IR
                        GraphOptimizer
                        gtest
                        TestMain)
add_glow_test(GraphOptzTest
              ${GLOW_BINARY_DIR}/tests/GraphOptzTest
                  --gtest_output=xml:GraphOptzTest.xml)

add_executable(GraphSchedulerTest
               GraphSchedulerTest.cpp)
target_link_libraries(GraphSchedulerTest
                      PRIVATE
                        Backend
                        Backends
                        Graph
                        IR
                        gtest
                        TestMain)
add_glow_test(GraphSchedulerTest
              ${GLOW_BINARY_DIR}/tests/GraphSchedulerTest
                  --gtest_output=xml:GraphSchedulerTest.xml)

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
add_glow_test(GraphTest
              ${GLOW_BINARY_DIR}/tests/GraphTest
                  --gtest_output=xml:GraphTest.xml)
LIST(APPEND UNOPT_TESTS ./tests/GraphTest -optimize-ir=false &&)

if(GLOW_WITH_HABANA)
  add_executable(HabanaGlowTest
                 HabanaGlowTest.cpp)
  target_link_libraries(HabanaGlowTest
                        PRIVATE
                          Backends
                          ExecutionEngine
                          Graph
                          GraphOptimizer
                          gtest
                          TestMain)
  target_include_directories(HabanaGlowTest PRIVATE "${SYNAPSE_INCLUDE_DIR}")
  add_glow_test(HabanaGlowTest ${GLOW_BINARY_DIR}/tests/HabanaGlowTest --gtest_output=xml:HabanaGlowTest.xml)

  add_executable(HabanaTest
          HabanaTest.cpp)
  target_link_libraries(HabanaTest
          PRIVATE
          Habana
          Support
          gtest
          TestMain)
  add_glow_test(HabanaTest
                ${GLOW_BINARY_DIR}/tests/HabanaTest
                    --gtest_output=xml:HabanaTest.xml)
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

  add_glow_test(HostManagerTest
                ${GLOW_BINARY_DIR}/tests/HostManagerTest
                    --gtest_output=xml:ProvisionerTest.xml)

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
  add_glow_test(HyphenTest
                ${GLOW_BINARY_DIR}/tests/HyphenTest
                    --gtest_output=xml:HyphenTest.xml)
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
                        GraphOptimizerPipeline
                        gtest
                        TestMain)
add_glow_test(IROptTest
              ${GLOW_BINARY_DIR}/tests/IROptTest
                  --gtest_output=xml:IROptTest.xml)

if(PNG_FOUND)
  add_executable(ImageTest
          ImageTest.cpp)
  target_link_libraries(ImageTest
          PRIVATE
          Base
          gtest
          TestMain)
  add_glow_test(ImageTest
                ${GLOW_BINARY_DIR}/tests/ImageTest
                    --gtest_output=xml:ImageTest.xml
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
  add_glow_test(LLVMIRGenTest
                ${GLOW_BINARY_DIR}/tests/LLVMIRGenTest
                    --gtest_output=xml:LLVMIRGenTest.xml)
endif()

add_executable(MemoryAllocatorTest
               MemoryAllocatorTest.cpp)
target_link_libraries(MemoryAllocatorTest
                      PRIVATE
                        CodeGen
                        gtest
                        Support
                        TestMain)
add_glow_test(MemoryAllocatorTest
              ${GLOW_BINARY_DIR}/tests/MemoryAllocatorTest
                  --gtest_output=xml:MemoryAllocatorTest.xml)

if(GLOW_WITH_OPENCL)
  add_executable(OCLTest
                 OCLTest.cpp)
  target_link_libraries(OCLTest
                        PUBLIC
                          OpenCL::OpenCL
                        PRIVATE
                          Backends
                          BackendTestUtils
                          ExecutionEngine
                          Graph
                          IR
                          gtest
                          TestMain)
  add_glow_test(OCLTest
                ${GLOW_BINARY_DIR}/tests/OCLTest
                    --gtest_output=xml:OCLTest.xml)
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
              COMMAND ${GLOW_BINARY_DIR}/tests/OnnxImporterTest
                  --gtest_output=xml:OnnxImporterTest.xml
              WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(OnnxExporterTest
               OnnxExporterTest.cpp)
target_link_libraries(OnnxExporterTest
                      PRIVATE
                        Backend
                        Graph
                        Importer
                        Exporter
                        ExecutionEngine
                        gtest
                        TestMain)
add_glow_test(NAME OnnxExporterTest
              COMMAND ${GLOW_BINARY_DIR}/tests/OnnxExporterTest
                  --gtest_output=xml:OnnxExporterTest.xml
              WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Loop through all backends present in lib/Backends (see FindBackends.cmake)
# and instantiate a backend-parameterized test for each of them.
foreach(backend ${GLOW_BACKENDS})
  add_backend_test(TEST BackendCorrectnessTest BACKEND "${backend}" UNOPT PRIVATE IR Backend Backends)
  add_backend_test(TEST BackendTest BACKEND "${backend}" UNOPT PRIVATE Backend HostManager)
  add_backend_test(TEST DeviceManagerTest BACKEND "${backend}")
  add_backend_test(TEST GradCheckTest BACKEND "${backend}" UNOPT)
  add_backend_test(TEST MLTest BACKEND "${backend}" UNOPT)
  add_backend_test(TEST OperatorGradTest BACKEND "${backend}" UNOPT)
  add_backend_test(TEST OperatorTest BACKEND "${backend}" UNOPT)
  add_backend_test(TEST TensorLayoutTest BACKEND "${backend}" UNOPT)
  add_backend_test(TEST
                   RecommendationSystemTest
                   BACKEND
                   "${backend}"
                   EXPENSIVE
                   PRIVATE
                   Partitioner
                   HostManager)
  add_backend_test(TEST TraceEventsTest BACKEND "${backend}" UNOPT)
  add_backend_test(TEST
                   DeferredWeightLoaderTest
                   BACKEND
                   "${backend}"
                   UNOPT
                   PRIVATE
                   HostManager
                   Runtime)

  add_backend_test(TEST
                   TypeAToTypeBFunctionConverterTest
                   BACKEND
                   "${backend}"
                   Converter
                   Graph)
endforeach()

add_executable(PartitionerTest
               PartitionerTest.cpp)
target_link_libraries(PartitionerTest
                      PRIVATE
                        Backend
                        Backends
                        ExecutionEngine
                        Graph
                        GraphOptimizer
                        IR
                        Partitioner
                        gtest
                        TestMain)

# There are multiple test cases in PartitionerTest that work only with CPU
# backend built-in.
if(GLOW_WITH_CPU)
  add_glow_test(PartitionerTest ${GLOW_BINARY_DIR}/tests/PartitionerTest --gtest_output=xml:PartitionerTest.xml)
endif()

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
  add_glow_test(ProvisionerTest
                ${GLOW_BINARY_DIR}/tests/ProvisionerTest
                    --gtest_output=xml:ProvisionerTest.xml)

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
add_glow_test(QuantizationTest
              ${GLOW_BINARY_DIR}/tests/QuantizationTest
                  --gtest_output=xml:QuantizationTest.xml)
LIST(APPEND UNOPT_TESTS ./tests/QuantizationTest -optimize-ir=false &&)

add_executable(StatsExporterTest
               StatsExporterTest.cpp)
target_link_libraries(StatsExporterTest
                      PRIVATE
                        Backends
                        Runtime
                        HostManager
                        gtest
                        TestMain
                        glog::glog)
add_glow_test(StatsExporterTest
              ${GLOW_BINARY_DIR}/tests/StatsExporterTest
                  --gtest_output=xml:StatsExporterTest.xml)

add_executable(TensorsTest
               TensorsTest.cpp)
target_link_libraries(TensorsTest
                      PRIVATE
                        Backend
                        Base
                        gtest
                        TestMain)
add_glow_test(TensorsTest
              ${GLOW_BINARY_DIR}/tests/TensorsTest
                  --gtest_output=xml:TensorsTest.xml)

add_executable(TensorPoolTest
               TensorPoolTest.cpp)
target_link_libraries(TensorPoolTest
                      PRIVATE
                        Backend
                        Backends
                        Graph
                        TensorPool
                        gtest
                        TestMain)
add_glow_test(TensorPoolTest
              ${GLOW_BINARY_DIR}/tests/TensorPoolTest
                  --gtest_output=xml:TensorPoolTest.xml)

add_executable(ThreadPoolTest
               ThreadPoolTest.cpp)
target_link_libraries(ThreadPoolTest
                      PRIVATE
                        Support
                        gtest
                        TestMain)
add_glow_test(ThreadPoolTest
              ${GLOW_BINARY_DIR}/tests/ThreadPoolTest
                  --gtest_output=xml:ThreadPoolTest.xml)

add_executable(UtilsTest
               StrCheck.cpp
               SupportTest.cpp
               TaggedListTest.cpp
               SharedMutexTest.cpp
               UtilsTest.cpp)
target_link_libraries(UtilsTest
                      PRIVATE
                        Base
                        Support
                        Testing
                        gtest
                        TestMain)
add_glow_test(NAME UtilsTest
              COMMAND ${GLOW_BINARY_DIR}/tests/UtilsTest
                  --gtest_output=xml:UtilsTest.xml
              WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(ErrorTest
              ErrorTest.cpp)
target_link_libraries(ErrorTest
                     PRIVATE
                       Support
                       gtest
                       TestMain)
add_glow_test(ErrorTest ${GLOW_BINARY_DIR}/tests/ErrorTest --gtest_output=xml:ErrorTest.xml)

LIST(APPEND UNOPT_TESTS true)
add_custom_target(test_unopt ${UNOPT_TESTS}
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
