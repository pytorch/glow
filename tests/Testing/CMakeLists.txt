add_library(Testing
              StrCheck.cpp)
target_link_libraries(Testing
                      PRIVATE
                        LLVMSupport
                        gtest)
