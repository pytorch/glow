
add_library(ExecutionEngine
              ExecutionEngine.cpp)

target_link_libraries(ExecutionEngine
                      PRIVATE
                        Interpreter
                        Optimizer
                        Base
                        Graph)

