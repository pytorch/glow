if(DEFINED ENV{TORCH_GLOW})
  SET(TORCH_GLOW_SRC $ENV{TORCH_GLOW})
  message(STATUS "Using TORCH_GLOW_SRC from env")
endif()

if(NOT EXISTS "${TORCH_GLOW}")
  message(FATAL_ERROR "No torch_glow directory found: " + ${TORCH_GLOW})
endif()

add_library(TorchGlowTraining
                     TorchGlowTraining.cpp)
target_compile_options(TorchGlowTraining
                      PRIVATE
                        -frtti -fexceptions -DC10_USE_GLOG)
target_link_libraries(TorchGlowTraining
                      PUBLIC
                        PyTorchFileLoader)

target_include_directories(TorchGlowTraining PUBLIC
                            ${TORCH_GLOW}/src
                            ${PYTORCH_DIR}/include)
