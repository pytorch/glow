add_library(GraphOptimizer
              GraphOptimizer.cpp
              Lower.cpp
              Quantization.cpp
              TrainingPreparation.cpp)

target_link_libraries(GraphOptimizer
                      PRIVATE
                        Backend
                        Converter
                        Graph
                        Quantization
                        QuantizationBase)
