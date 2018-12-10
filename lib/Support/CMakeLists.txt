add_library(Support
              Debug.cpp
              Error.cpp
              Random.cpp
              Support.cpp)
target_link_libraries(Support
                      PRIVATE
                        LLVMSupport)

add_library(ThreadPool
              ThreadPool.cpp)


