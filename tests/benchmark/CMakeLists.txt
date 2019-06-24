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

add_executable(RuntimeBench
               RuntimeBench.cpp)
target_include_directories(RuntimeBench
                           PRIVATE
                             ${CMAKE_SOURCE_DIR}/lib/Backends/CPU
                             ${CMAKE_SOURCE_DIR}/lib/Backends/Interpreter)
target_link_libraries(RuntimeBench
                      PRIVATE
                        Backend
                        Backends
                        ExecutionEngine
                        Executor
                        HostManager
                        Graph
                        GraphOptimizer
                        benchmark)
endif()
