
add_executable(cifar10
                 cifar10.cpp)
target_link_libraries(cifar10
                      PRIVATE
                        Interpreter
                        Network
                        IR
                        Support)

add_executable(mnist
                 mnist.cpp)
target_link_libraries(mnist
                      PRIVATE
                        Interpreter
                        Network
                        IR
                        Support)

