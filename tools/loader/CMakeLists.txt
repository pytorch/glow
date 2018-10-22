add_executable(image-classifier
  Loader.cpp
  ImageClassifier.cpp)

target_link_libraries(image-classifier
                      PRIVATE
                        Base
                        Converter
                        ExecutionEngine
                        Graph
                        Importer
                        Optimizer
                        Quantization)

add_executable(text-translator
  Loader.cpp
  TextTranslator.cpp)

target_link_libraries(text-translator
                      PRIVATE
                        Base
                        Converter
                        Graph
                        Importer
                        ExecutionEngine
                        Optimizer
                        Quantization)

add_executable(model-runner
  Loader.cpp
  ModelRunner.cpp)

target_link_libraries(model-runner
                      PRIVATE
                        Base
                        Converter
                        Graph
                        Importer
                        ExecutionEngine
                        Optimizer
                        Quantization)
