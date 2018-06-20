
add_library(Support
              Debug.cpp
              Random.cpp
              Support.cpp)
target_link_libraries(Support
                      INTERFACE
                        LLVMSupport)

