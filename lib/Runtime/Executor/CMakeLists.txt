add_library(Executor
              ThreadPoolExecutor.cpp)

target_link_libraries(Executor
                      PRIVATE
                        ExecutionContext
                        Graph)
