add_library(Executor
              Executor.cpp
              ThreadPoolExecutor.cpp)

target_link_libraries(Executor
                      PUBLIC
                        ThreadPool
                      PRIVATE
                        ExecutionContext
                        Graph)
