
add_library(Optimizer
            IROptimizer.cpp
            GraphOptimizer.cpp
            Lower.cpp
            OptimizerUtils.cpp
            Quantization.cpp)

target_link_libraries(Optimizer
                      PRIVATE
                        IR)

