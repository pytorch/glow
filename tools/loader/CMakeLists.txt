
add_executable(image-classifier
  Loader.cpp
  ImageClassifier.cpp)

target_link_libraries(image-classifier
                      PRIVATE
                        Base
                        Importer
                        ExecutionEngine
                        IR
                        Quantization)

add_executable(model-runner
  Loader.cpp
  ModelRunner.cpp)

target_link_libraries(model-runner
                      PRIVATE
                        Base
                        Importer
                        ExecutionEngine
                        IR
                        Quantization)
