set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOW_BINARY_DIR}/tests/stress/)

add_executable(SparseLengthsSumTest
  SparseLengthsSumTest.cpp)
target_link_libraries(SparseLengthsSumTest
  PRIVATE
  Backends
  BackendTestUtils
  ExecutionEngine
  gtest
  glog::glog)

add_executable(ParameterSweepTest
               ParameterSweepTest.cpp)
target_link_libraries(ParameterSweepTest
                      PRIVATE
                        Backends
                        BackendTestUtils
                        ExecutionEngine
                        glog::glog
                        gtest
                        Graph
                        GraphOptimizer
                        Support)
