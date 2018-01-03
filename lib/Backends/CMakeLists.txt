
add_subdirectory(Interpreter)
add_subdirectory(OpenCL)
add_subdirectory(JIT)

add_library(Backends Backends.cpp)
target_link_libraries(Backends
                      PRIVATE
                        Interpreter
                        OpenCL
                        JIT
                        Base
                        Graph)
