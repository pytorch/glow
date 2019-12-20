add_library(Lower
              Lower.cpp)

target_link_libraries(Lower
                      PRIVATE
                        Graph
                        Quantization
                        QuantizationBase)
