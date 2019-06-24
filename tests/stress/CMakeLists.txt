set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOW_BINARY_DIR}/tests/stress/)

add_executable(SparseLengthsSumTest
  SparseLengthsSumTest.cpp)
target_link_libraries(SparseLengthsSumTest
  PRIVATE
  Backends
  ExecutionEngine
  gtest
  glog::glog)

add_executable(ParameterSweepTest
               ParameterSweepTest.cpp)
target_link_libraries(ParameterSweepTest
                      PRIVATE
                        BackendTestUtils
                        ExecutionEngine
                        glog::glog
                        gtest
                        Graph
                        GraphOptimizer
                        Support)
