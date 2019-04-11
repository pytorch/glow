add_library(QuantizationBase
              Profile.cpp
              Base.cpp)
target_link_libraries(QuantizationBase
  PRIVATE
  LLVMSupport)

add_dependencies(QuantizationBase AutoGen)
