add_library(Support
              Debug.cpp
              Error.cpp
              Random.cpp
              Support.cpp)
target_link_libraries(Support
                      INTERFACE
                        LLVMSupport)
