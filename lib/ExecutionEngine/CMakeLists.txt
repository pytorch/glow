add_library(ExecutionEngine
              ExecutionEngine.cpp)

target_link_libraries(ExecutionEngine
                      PRIVATE
                        Backend
                        Backends
                        GraphOptimizer
                        Base
                        Graph)
