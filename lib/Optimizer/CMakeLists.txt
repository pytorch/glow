add_library(Optimizer
              IROptimizer.cpp
              GraphOptimizer.cpp
              Lower.cpp
              Quantization.cpp
              TrainingPreparation.cpp)

target_link_libraries(Optimizer
                      PRIVATE
                        Backend
                        Converter
                        Graph
                        IR
                        Quantization)
