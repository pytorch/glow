add_subdirectory(Base)

add_library(Quantization
              Serialization.cpp
              Quantization.cpp)

target_link_libraries(Quantization
                      PRIVATE
                        Graph
                        ExecutionEngine
                        QuantizationBase)
