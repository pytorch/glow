add_library(ExecutorTestBuilder
            ExecutorTestBuilder.cpp)
target_link_libraries(ExecutorTestBuilder
                      PRIVATE
                        benchmark
                        Base
                        DeviceManager
                        ExecutorUnitTestWrapper
                        Graph
                        TestDeviceManager)

add_library(ExecutorBenchmarkWrapper
            ExecutorBenchmarkWrapper.cpp)
target_link_libraries(ExecutorBenchmarkWrapper
                      PRIVATE
                        benchmark
                        Base
                        Executor
                        Graph)

add_library(ExecutorUnitTestWrapper
            ExecutorUnitTestWrapper.cpp)
target_link_libraries(ExecutorUnitTestWrapper
                      PRIVATE
                        Base
                        Executor
                        Graph)

add_library(TestDeviceManager
            TestDeviceManager.cpp)
target_link_libraries(TestDeviceManager
                      PRIVATE
                        Base
                        DeviceManager
                        Graph
                        ThreadPool)
