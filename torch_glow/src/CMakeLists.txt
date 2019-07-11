if(DEFINED ENV{PYTORCH_DIR})
  SET(PYTORCH_DIR $ENV{PYTORCH_DIR})
  message(STATUS "Using PYTORCH_DIR from env")
endif()

if(NOT EXISTS "${PYTORCH_DIR}")
  message(FATAL_ERROR "No PyTorch installation found")
endif()

message(STATUS "Using pytorch dir ${PYTORCH_DIR}")

link_directories(${PYTORCH_DIR}/lib)

pybind11_add_module(_torch_glow
                      binding.cpp
                      PyTorchModelLoader.cpp
                      CachingGraphRunner.cpp)

target_compile_options(_torch_glow PRIVATE -frtti -fexceptions)
target_link_libraries(_torch_glow
                      PRIVATE
                        torch
                        c10
                        Support
                        ExecutionEngine
                        Graph
                        Importer
                        Support
                        Backends
                        pybind11)

target_include_directories(_torch_glow PUBLIC
                            ${PYTORCH_DIR}/include
)
