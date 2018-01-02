
add_library(CodeGen MemoryAllocator.cpp)
target_link_libraries(CodeGen
                      PRIVATE
                        Base)
