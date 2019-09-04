add_library(Executor
              ExecutionState.cpp
              ThreadPoolExecutor.cpp)

target_link_libraries(Executor
                      PRIVATE
                        ExecutionContext
                        Graph)
