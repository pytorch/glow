add_library(IROptimizer
              IROptimizer.cpp)

target_link_libraries(IROptimizer
                      PRIVATE
                        Graph
                        IR
                        QuantizationBase)
