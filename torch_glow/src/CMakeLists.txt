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
                     CachingGraphRunner.cpp
                     CustomPyTorchOpLoader.cpp
                     GlowFuser.cpp
                     FuseKnownPatterns.cpp
                     GlowIValue.cpp
                     PyTorchCommon.cpp
                     Registration.cpp
                     PyTorchModelLoader.cpp)
target_compile_options(PyTorchModelLoader
                      PRIVATE
                        -frtti -fexceptions -DC10_USE_GLOG)
target_link_libraries(PyTorchModelLoader
                      PRIVATE
                        torch_cpu
                        c10
                        Support
                        HostManager
                        Graph
                        Importer)

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
                        TorchGlowTraining
                        Backends
                        pybind11)

target_include_directories(PyTorchModelLoader PUBLIC
                            ${PYTORCH_DIR}/include)

target_include_directories(PyTorchFileLoader PUBLIC
                            ${PYTORCH_DIR}/include)

target_include_directories(_torch_glow PUBLIC
                            ${PYTORCH_DIR}/include
                            ${TORCH_GLOW}/src/training)

add_subdirectory(training)
