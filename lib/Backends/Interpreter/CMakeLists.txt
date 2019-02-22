add_library(Interpreter
              Interpreter.cpp
              InterpreterFunction.cpp
              InterpreterNodes.cpp)
target_link_libraries(Interpreter
                      PRIVATE
                        Backend
                        Base
                        Graph
                        CodeGen
                        IR
                        Optimizer
                        QuantizationBase)

add_library(InterpreterDeviceManager
              InterpreterDeviceManager.cpp)
target_link_libraries(InterpreterDeviceManager
                      PRIVATE
                        Backend
                        Interpreter
                        Base
                        CodeGen
                        Graph
                        IR
                        Optimizer
                        ThreadPool)

add_library(InterpreterFactory
              InterpreterFactory.cpp)
target_link_libraries(InterpreterFactory
                      PRIVATE
                        Interpreter)

