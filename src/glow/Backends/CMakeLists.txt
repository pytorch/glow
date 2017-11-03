
add_subdirectory(Interpreter)

add_library(Backends Backends.cpp)
target_link_libraries(Backends
                      PRIVATE
                        Interpreter
                        Base
                        Graph)
