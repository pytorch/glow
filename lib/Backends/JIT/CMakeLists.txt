
add_library(JIT
            JIT.cpp)
target_link_libraries(JIT
                      PRIVATE
                        Optimizer
                        Base
                        Graph
                        LLVMCore)

