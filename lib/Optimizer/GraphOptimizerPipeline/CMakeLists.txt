add_library(GraphOptimizerPipeline
              FunctionPassPipeline.cpp)

target_link_libraries(GraphOptimizerPipeline
                      PRIVATE
                        Graph
                        LLVMCore
                        PassManager)
