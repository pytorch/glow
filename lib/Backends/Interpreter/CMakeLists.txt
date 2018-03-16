
add_library(Interpreter
              Interpreter.cpp
              InterpreterNodes.cpp)
target_link_libraries(Interpreter
                      PRIVATE
                        Base
                        Graph
                        Quantization)
