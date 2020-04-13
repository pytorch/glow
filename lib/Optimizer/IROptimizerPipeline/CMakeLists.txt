add_library(IROptimizerPipeline
              IRFunctionPassPipeline.cpp)

target_link_libraries(IROptimizerPipeline
                      PRIVATE
                        IR
                        LLVMCore
                        PassManager)
