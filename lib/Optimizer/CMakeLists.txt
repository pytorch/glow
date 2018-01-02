
add_library(Optimizer
            IROptimizer.cpp
            GraphOptimizer.cpp
            Lower.cpp)

target_link_libraries(Optimizer
                      PRIVATE
                        IR)

