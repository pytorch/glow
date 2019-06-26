add_library(Interpreter
              Interpreter.cpp
              InterpreterFunction.cpp
              InterpreterNodes.cpp
              InterpreterDeviceManager.cpp
              InterpreterFactory.cpp)
target_link_libraries(Interpreter
                      PRIVATE
                        Backend
                        Base
                        CodeGen
                        Graph
                        IR
                        LLVMCore
                        IROptimizer
                        QuantizationBase)

set(linked_backends ${linked_backends} Interpreter PARENT_SCOPE)
