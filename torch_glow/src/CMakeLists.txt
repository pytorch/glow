if(DEFINED ENV{PYTORCH_DIR})
  SET(PYTORCH_DIR $ENV{PYTORCH_DIR})
  message(STATUS "Using PYTORCH_DIR from env")
endif()

if(NOT EXISTS "${PYTORCH_DIR}")
  message(FATAL_ERROR "No PyTorch installation found")
endif()

message(STATUS "Using pytorch dir ${PYTORCH_DIR}")

link_directories(${PYTORCH_DIR}/lib)

add_library(PyTorchModelLoader
                     PyTorchCommon.cpp
                     GlowFuser.cpp
                     FuseLinear.cpp
                     PyTorchModelLoader.cpp
                     CachingGraphRunner.cpp)
target_compile_options(PyTorchModelLoader
                      PRIVATE
                        -frtti -fexceptions -DC10_USE_GLOG)
target_link_libraries(PyTorchModelLoader
                      PRIVATE
                        torch
                        c10
                        Support
                        ExecutionEngine
                        Graph
                        Importer
                        Backends)

add_library(PyTorchFileLoader
                     PyTorchFileLoader.cpp)
target_compile_options(PyTorchFileLoader
                      PRIVATE
                        -frtti -fexceptions -DC10_USE_GLOG)
target_link_libraries(PyTorchFileLoader
                      PRIVATE
                        PyTorchModelLoader)

pybind11_add_module(_torch_glow
                      binding.cpp)

target_compile_options(_torch_glow
                      PRIVATE
                        -frtti -fexceptions -DC10_USE_GLOG)
target_link_libraries(_torch_glow
                      PRIVATE
                        PyTorchModelLoader
                        pybind11)

target_include_directories(PyTorchModelLoader PUBLIC
                            ${PYTORCH_DIR}/include)

target_include_directories(PyTorchFileLoader PUBLIC
                            ${PYTORCH_DIR}/include)

target_include_directories(_torch_glow PUBLIC
                            ${PYTORCH_DIR}/include)

add_subdirectory(training)
