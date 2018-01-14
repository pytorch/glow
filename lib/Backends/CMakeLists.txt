
add_library(Backends Backends.cpp)

add_subdirectory(Interpreter)

if(OpenCL_FOUND)
  target_compile_definitions(Backends
                             PRIVATE
                             WITH_OPENCL=1)
  add_subdirectory(OpenCL)
  LIST(APPEND linked_backends OpenCL)
endif()

add_subdirectory(JIT)
LIST(APPEND linked_backends JIT)

target_link_libraries(Backends
                      PRIVATE
                        Interpreter
                        ${linked_backends} 
                        Base
                        Graph
                        CodeGen)
