add_library(ExecutionEngine
              ExecutionEngine.cpp)

target_link_libraries(ExecutionEngine
                      PRIVATE
                        Backends
                        Optimizer
                        Base
                        Graph
                        IR)
