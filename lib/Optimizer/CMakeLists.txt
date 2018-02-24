
add_library(Optimizer
            IROptimizer.cpp
            GraphOptimizer.cpp
            Lower.cpp
            Quantization.cpp)

target_link_libraries(Optimizer
                      PRIVATE
                        Quantization
                        IR)

