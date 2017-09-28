
add_library(Support
              Random.cpp
              Support.cpp)
target_link_libraries(Support
                      INTERFACE
                        LLVMSupport)

