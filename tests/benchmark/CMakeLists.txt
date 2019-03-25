if(GLOW_WITH_CPU AND NOT MSVC)
add_executable(GemmBench
               GemmBench.cpp)
target_link_libraries(GemmBench
                      PRIVATE
                        CPURuntimeNative)

add_executable(ConvBench
               ConvBench.cpp)
target_link_libraries(ConvBench
                      PRIVATE
                        CPURuntimeNative)

add_executable(DeviceManagerBench
               DeviceManagerBench.cpp)
target_link_libraries(DeviceManagerBench
                      PRIVATE
                        Backend
                        Backends
                        DeviceManager
                        Graph
                        IR
                        ExecutionEngine
                        Optimizer
                        benchmark)
endif()
