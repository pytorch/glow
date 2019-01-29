add_executable(cifar10
                 cifar10.cpp)
target_link_libraries(cifar10
                      PRIVATE
                        Backends
                        ExecutionEngine
                        Graph
                        IR
                        Support
                        LLVMSupport)

add_executable(mnist
                 mnist.cpp)
target_link_libraries(mnist
                      PRIVATE
                        Backends
                        ExecutionEngine
                        Graph
                        IR
                        Optimizer
                        Support
                        LLVMSupport)

add_executable(ptb
                 ptb.cpp)
target_link_libraries(ptb
                      PRIVATE
                        Backends
                        ExecutionEngine
                        Graph
                        IR
                        Support
                        LLVMSupport)

add_executable(char-rnn
                 char-rnn.cpp)
target_link_libraries(char-rnn
                      PRIVATE
                        Backends
                        ExecutionEngine
                        Graph
                        IR
                        Optimizer
                        Support)

add_executable(fr2en
                 fr2en.cpp)
target_link_libraries(fr2en
                      PRIVATE
                        Base
                        ExecutionEngine
                        IR
                        Optimizer
                        Quantization
                        Support
                        LLVMSupport)

add_executable(lenet-loader
                 lenet-loader.cpp)
target_link_libraries(lenet-loader
                      PRIVATE
                        ExecutionEngine
                        Graph
                        Importer
                        Support)

if(GLOW_WITH_CPU)
  add_executable(resnet-verify
                   resnet-verify.cpp)
  target_link_libraries(resnet-verify
                        PRIVATE
                          ExecutionEngine
                          Graph
                          Importer)

  add_executable(resnet-concurrent
                   resnet-concurrent.cpp)
  target_link_libraries(resnet-concurrent
                        PRIVATE
                          Backends
                          CPUDeviceManager
                          DeviceManager
                          ExecutionEngine
                          Graph
                          Importer
                          Optimizer)
  target_include_directories(resnet-concurrent PUBLIC ${CMAKE_SOURCE_DIR}/lib)
endif()

if(GLOW_WITH_BUNDLES)
  add_subdirectory(bundles)
endif()
