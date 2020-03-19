add_library(Executor
              NetworkExecutionState.cpp
              ThreadPoolExecutor.cpp)

target_link_libraries(Executor
                      PRIVATE
                        Backend
                        Backends
                        ExecutionContext
                        Graph)
