
add_library(Optimizer
              ${CMAKE_SOURCE_DIR}/include/glow/Optimizer/Optimizer.h

              ${CMAKE_SOURCE_DIR}/src/glow/Optimizer/IROptimizer.cpp
              ${CMAKE_SOURCE_DIR}/src/glow/Optimizer/GraphOptimizer.cpp)
target_link_libraries(Optimizer
                      PRIVATE
                        IR)
