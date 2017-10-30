
add_library(ExecutionEngine
              ${CMAKE_SOURCE_DIR}/include/glow/ExecutionEngine/ExecutionEngine.h

              ${CMAKE_SOURCE_DIR}/src/glow/ExecutionEngine/ExecutionEngine.cpp)
target_link_libraries(ExecutionEngine
                      PRIVATE
                        Interpreter
                        Optimizer
                        Base
                        Graph)
