if(DEFINED ENV{PYTORCH_DIR})
  SET(PYTORCH_DIR $ENV{PYTORCH_DIR})
  message(STATUS "Using PYTORCH_DIR from env")
endif()

if(NOT EXISTS "${PYTORCH_DIR}")
  message(FATAL_ERROR "No PyTorch installation found")
endif()

message(STATUS "Using pytorch dir ${PYTORCH_DIR}")

if(DEFINED ENV{TORCH_GLOW})
  SET(TORCH_GLOW_SRC $ENV{TORCH_GLOW})
  message(STATUS "Using TORCH_GLOW_SRC from env")
endif()

if(NOT EXISTS "${TORCH_GLOW}")
  message(FATAL_ERROR "No torch_glow directory found: " + ${TORCH_GLOW})
endif()

link_directories(${PYTORCH_DIR}/lib)

add_executable(PyTorchLoaderTest
               ${TORCH_GLOW}/tests/unittests/PyTorchLoaderTest.cpp)
target_compile_options(PyTorchLoaderTest
                      PRIVATE
                        -frtti -fexceptions -DC10_USE_GLOG)
target_link_libraries(PyTorchLoaderTest
                      PRIVATE
                        PyTorchFileLoader
                        gtest
                        TestMain)

target_include_directories(PyTorchLoaderTest PUBLIC
                            ${PYTORCH_DIR}/include
                            ${TORCH_GLOW}/src)
