if(GLOW_WITH_CPU AND NOT MSVC)
add_executable(AddBench
               AddBench.cpp)
target_link_libraries(AddBench
                      PRIVATE
                        ExecutionEngine
                        Graph
                        HostManager
                        CPURuntimeNative)

add_executable(SLSBench
               SLSBench.cpp)
target_link_libraries(SLSBench
                      PRIVATE
                        ExecutionEngine
                        Graph
                        HostManager
                        CPURuntimeNative)

add_executable(ConvBench
               ConvBench.cpp)
target_link_libraries(ConvBench
                      PRIVATE
                        ExecutionEngine
                        CPURuntimeNative)

add_executable(GemmBench
               GemmBench.cpp)
target_link_libraries(GemmBench
                      PRIVATE
                        ExecutionEngine
                        Graph
                        GraphOptimizer
                        HostManager
                        CPURuntimeNative)

add_executable(GemmParallelBench
               GemmParallelBench.cpp)
target_link_libraries(GemmParallelBench
                      PRIVATE
                        ExecutionEngine
                        Graph
                        GraphOptimizer
                        HostManager
                        CPURuntimeNative)

add_executable(BatchGemmBench
               BatchGemmBench.cpp)
target_link_libraries(BatchGemmBench
                      PRIVATE
                        ExecutionEngine
                        Graph
                        GraphOptimizer
                        HostManager
                        CPURuntimeNative)

add_executable(TransposeBench
               TransposeBench.cpp)
target_link_libraries(TransposeBench
                      PRIVATE
                        ExecutionEngine
                        Graph
                        GraphOptimizer
                        HostManager
                        CPURuntimeNative)

add_executable(RuntimeBench
               RuntimeBench.cpp)
target_include_directories(RuntimeBench
                           PRIVATE
                             ${GLOW_SOURCE_DIR}/lib/Backends/CPU
                             ${GLOW_SOURCE_DIR}/lib/Backends/Interpreter)
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
