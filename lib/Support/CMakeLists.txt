
add_library(Support
              Random.cpp
              Support.cpp)
target_link_libraries(Support
                      INTERFACE
                        LLVMSupport)
target_include_directories(Support
                           PUBLIC
                             $<TARGET_PROPERTY:LLVMSupport,INTERFACE_INCLUDE_DIRECTORIES>)
add_dependencies(Support llvm-install)

