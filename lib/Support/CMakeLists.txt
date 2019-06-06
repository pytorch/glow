add_library(Support
              Debug.cpp
              Error.cpp
              Random.cpp
              Support.cpp
              ThreadPool.cpp)
target_link_libraries(Support
                      PUBLIC
                        LLVMSupport
                        glog::glog)

add_subdirectory(TensorPool)
