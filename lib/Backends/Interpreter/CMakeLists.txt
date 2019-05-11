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
                        Optimizer
                        QuantizationBase)

set(linked_factories ${linked_factories} Interpreter PARENT_SCOPE)
set(linked_device_managers ${linked_device_managers} Interpreter PARENT_SCOPE)
