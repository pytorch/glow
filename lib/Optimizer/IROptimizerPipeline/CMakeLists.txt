add_library(IROptimizerPipeline
              IRFunctionPassPipeline.cpp
              CommandLine.cpp)

target_link_libraries(IROptimizerPipeline
                      PRIVATE
                        IR
                        LLVMCore
                        PassManager)
