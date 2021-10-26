if(DEFINED ENV{PYTORCH_DIR})
  SET(PYTORCH_DIR $ENV{PYTORCH_DIR})
  message(STATUS "Using PYTORCH_DIR from env")
endif()

if(NOT EXISTS "${PYTORCH_DIR}")
  message(FATAL_ERROR "No PyTorch installation found")
endif()

message(STATUS "Using pytorch dir ${PYTORCH_DIR}")

link_directories(${PYTORCH_DIR}/lib)

add_library(PyTorchShapeInference
                     ShapeInferenceEngine.cpp)
target_compile_options(PyTorchShapeInference
                      PRIVATE
                        -frtti -fexceptions -DC10_USE_GLOG)
target_link_libraries(PyTorchShapeInference
                      PRIVATE
                        torch_cpu
                        c10
                        folly_jemalloc)

add_library(PyTorchModelLoader
                     CachingGraphRunner.cpp
                     CustomPyTorchOpLoader.cpp
                     GlowFuser.cpp
                     FuseKnownPatterns.cpp
                     GlowIValue.cpp
                     InputMeta.cpp
                     PyTorchCommon.cpp
                     Registration.cpp
                     PyTorchModelLoader.cpp
                     TorchGlowBackend.cpp
                     GlowCompileSpec.cpp)
target_compile_options(PyTorchModelLoader
                      PRIVATE
                        -frtti -fexceptions -DC10_USE_GLOG)
target_link_libraries(PyTorchModelLoader
                      PRIVATE
                        torch_cpu
                        c10)
target_link_libraries(PyTorchModelLoader
                      PUBLIC
                        Backends
                        ExecutionContext
                        ExecutionEngine
                        Exporter
                        Graph
                        HostManager
                        Importer
                        GraphOptimizer
                        Quantization
                        Runtime
                        PyTorchShapeInference)

pybind11_add_module(_torch_glow
                      binding.cpp)

target_compile_options(_torch_glow
                      PRIVATE
                        -frtti -fexceptions -DC10_USE_GLOG)
target_link_libraries(_torch_glow
                      PRIVATE
                        PyTorchModelLoader
                        Backends
                        torch_python)

include_directories(${PYTORCH_DIR}/include)
