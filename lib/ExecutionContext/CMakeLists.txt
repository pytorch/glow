add_library(ExecutionContext
              TraceEvents.cpp)
target_link_libraries(ExecutionContext
                      PRIVATE
                        Base
                        Graph)

