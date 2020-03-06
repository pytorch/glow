add_library(Executor
              NetworkExecutionState.cpp
              ThreadPoolExecutor.cpp)

target_link_libraries(Executor
                      PRIVATE
                        ExecutionContext
                        Graph)
