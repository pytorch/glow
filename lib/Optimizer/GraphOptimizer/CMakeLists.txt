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
                        Converter
                        Graph
                        GraphOptimizerPipeline
                        Lower
                        Interpreter
                        Quantization
                        QuantizationBase)
