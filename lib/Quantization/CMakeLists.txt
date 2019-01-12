add_subdirectory(Base)

find_package(glog REQUIRED)

add_library(Quantization
              Serialization.cpp
              Quantization.cpp)

target_link_libraries(Quantization
                      PRIVATE
                        Converter
                        Graph
                        ExecutionEngine
                        QuantizationBase
                        LLVMSupport
                        glog::glog)
