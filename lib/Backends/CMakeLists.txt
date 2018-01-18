add_library(Backends Backends.cpp)

add_subdirectory(Interpreter)

if(GLOW_WITH_OPENCL)
  target_compile_definitions(Backends PRIVATE GLOW_WITH_OPENCL=1)
  add_subdirectory(OpenCL)
  LIST(APPEND linked_backends OpenCL)
endif()


if(GLOW_WITH_JIT)
  target_compile_definitions(Backends PRIVATE GLOW_WITH_JIT=1)
  add_subdirectory(JIT)
  LIST(APPEND linked_backends JIT)
endif()

target_link_libraries(Backends
                      PRIVATE
                        Interpreter
                        ${linked_backends} 
                        Base
                        Graph
                        CodeGen)
