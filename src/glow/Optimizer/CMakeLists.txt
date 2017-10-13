
add_library(Optimizer
            IROptimizer.cpp
            GraphOptimizer.cpp)

target_link_libraries(Optimizer
                      PRIVATE
                        IR)

