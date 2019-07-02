add_library(GraphOptimizerPipeline
              Pipeline.cpp)

target_link_libraries(GraphOptimizerPipeline
                      PRIVATE
                        Graph
                        LLVMCore)
