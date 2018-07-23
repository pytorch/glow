
add_library(Optimizer
              IROptimizer.cpp
              GraphOptimizer.cpp
              Lower.cpp
              Partition.cpp
              Quantization.cpp)

target_link_libraries(Optimizer
                      PRIVATE
                        Graph
                        IR
                        Backends
                        QuantizationBase)
