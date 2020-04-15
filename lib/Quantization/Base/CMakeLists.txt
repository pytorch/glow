add_library(QuantizationBase
              Profile.cpp
              Base.cpp
              Calibration.cpp)
target_link_libraries(QuantizationBase
  PRIVATE
  LLVMSupport
  Base)

add_dependencies(QuantizationBase AutoGen)
