add_library(Executor
              Executor.cpp
              ThreadPoolExecutor.cpp)

target_link_libraries(Executor
	              PRIVATE
		        Graph)
