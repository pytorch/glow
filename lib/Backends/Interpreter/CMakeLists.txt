
add_library(Interpreter
              Interpreter.cpp
              InterpreterNodes.cpp
              Quantization/Quantization.cpp)
target_link_libraries(Interpreter
                      PRIVATE
                        Optimizer
                        Base
                        Graph)

