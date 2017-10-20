
add_library(Interpreter
              ${CMAKE_SOURCE_DIR}/include/glow/Interpreter/Interpreter.h

              ${CMAKE_SOURCE_DIR}/src/glow/Interpreter/Interpreter.cpp
              ${CMAKE_SOURCE_DIR}/src/glow/Interpreter/InterpreterNodes.cpp)
target_link_libraries(Interpreter
                      PRIVATE
                        Optimizer
                        Base
                        Graph)
