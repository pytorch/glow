add_library(ExecutionEngine
              ExecutionEngine.cpp)

target_link_libraries(ExecutionEngine
                      PRIVATE
                        Backend
                        Backends
                        GraphOptimizer
                        Base
                        Graph)


add_library(ExecutionEngine2
              ExecutionEngine2.cpp)

target_link_libraries(ExecutionEngine2
                      PRIVATE
                        Backend
                        Backends
                        HostManager
                        GraphOptimizer
                        Base
                        Graph)
