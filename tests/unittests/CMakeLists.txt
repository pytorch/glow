
add_executable(tensorsTest
                 Tensors.cpp)
target_link_libraries(tensorsTest
                      PRIVATE
                        Support
                        gtest
                        gtest_main)
add_test(NAME
          tensorsTest
        COMMAND
          $<TARGET_FILE:tensorsTest>)

add_executable(IRgradCheckTest
                 IRGradCheck.cpp)
target_link_libraries(IRgradCheckTest
                      PRIVATE
                        Interpreter
                        ExecutionEngine
                        IR
                        gtest
                        gtest_main)
add_test(NAME
           IRgradCheckTest
         COMMAND
           $<TARGET_FILE:IRgradCheckTest>)

add_executable(IRTest
                 IRTest.cpp)
target_link_libraries(IRTest
                      PRIVATE
                        Interpreter
                        IR
                        gtest
                        gtest_main)
add_test(NAME
           IRTest
         COMMAND
           $<TARGET_FILE:IRTest>)

add_executable(InterpreterTest
                 InterpreterTest.cpp)
target_link_libraries(InterpreterTest
                      PRIVATE
                        Interpreter
                        IR
                        ExecutionEngine
                        gtest
                        gtest_main)
add_test(NAME
           InterpreterTest
         COMMAND
           $<TARGET_FILE:InterpreterTest>)

add_executable(GraphTest
                 GraphTest.cpp)
target_link_libraries(GraphTest
                      PRIVATE
                        Graph
                        IR
                        Interpreter
                        gtest
                        gtest_main)
add_test(NAME
           GraphTest
         COMMAND
           $<TARGET_FILE:GraphTest>)


add_executable(GraphOptz
                 GraphOpt.cpp)
target_link_libraries(GraphOptz
                      PRIVATE
                        Graph
                        IR
                        Interpreter
                        gtest
                        gtest_main)
add_test(NAME
           GraphOptz
         COMMAND
            $<TARGET_FILE:GraphOptz>)

