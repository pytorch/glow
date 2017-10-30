
add_library(Support
              ${CMAKE_SOURCE_DIR}/include/glow/Support/Compiler.h
              ${CMAKE_SOURCE_DIR}/include/glow/Support/Random.h
              ${CMAKE_SOURCE_DIR}/include/glow/Support/Support.h

              ${CMAKE_SOURCE_DIR}/src/glow/Support/Random.cpp
              ${CMAKE_SOURCE_DIR}/src/glow/Support/Support.cpp)
target_link_libraries(Support
                      INTERFACE
                        LLVMSupport)
target_include_directories(Support
                           PUBLIC
                             $<TARGET_PROPERTY:LLVMSupport,INTERFACE_INCLUDE_DIRECTORIES>)
add_dependencies(Support llvm-install)
