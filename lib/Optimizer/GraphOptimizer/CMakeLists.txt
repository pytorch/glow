add_library(GraphOptimizer
              ConstantFolding.cpp
              GraphOptimizer.cpp
              Lower.cpp
              PassManager.cpp
              Quantization.cpp
              TrainingPreparation.cpp)

target_link_libraries(GraphOptimizer
                      PRIVATE
                        Backend
                        Backends
                        Converter
                        Graph
                        GraphOptimizerPipeline
                        Lower
                        Quantization
                        QuantizationBase)
