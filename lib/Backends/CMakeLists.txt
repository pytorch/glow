add_library(Backends Backends.cpp)

add_subdirectory(Interpreter)

if(GLOW_WITH_OPENCL)
  target_compile_definitions(Backends PRIVATE GLOW_WITH_OPENCL=1)
  add_subdirectory(OpenCL)
  LIST(APPEND linked_backends OpenCL)
endif()


if(GLOW_WITH_CPU)
  target_compile_definitions(Backends PRIVATE GLOW_WITH_CPU=1)
  add_subdirectory(CPU)
  LIST(APPEND linked_backends CPUBackend)
endif()

target_link_libraries(Backends
                      PRIVATE
                        Interpreter
                        ${linked_backends} 
                        Base
                        Graph
                        CodeGen)
