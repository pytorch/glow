add_library(IROptimizer
              IRFunctionPassManager.cpp
              IROptimizer.cpp
              IRInstrumentation.cpp)

target_link_libraries(IROptimizer
                      PRIVATE
                        Graph
                        IR
                        QuantizationBase
                        PassManager)
