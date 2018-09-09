add_library(ExecutionEngine
              Context.cpp
              ExecutionEngine.cpp)

target_link_libraries(ExecutionEngine
                      PRIVATE
                        Backends
                        Optimizer
                        Base
                        Graph
                        IR)
