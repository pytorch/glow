add_library(IROptimizer
              IRFunctionPassManager.cpp
              IROptimizer.cpp)

target_link_libraries(IROptimizer
                      PRIVATE
                        Graph
                        IR
                        QuantizationBase
                        PassManager)
