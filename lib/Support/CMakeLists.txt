add_library(Support
              Debug.cpp
              Error.cpp
              Random.cpp
              Support.cpp
              ThreadPool.cpp
              ZipUtils.cpp)
target_link_libraries(Support
                      PUBLIC
                        LLVMSupport
                        Miniz
                        glog::glog)

add_subdirectory(TensorPool)
