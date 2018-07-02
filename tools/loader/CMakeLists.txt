
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

target_include_directories(image-classifier
                           PRIVATE
                           ${PROTOBUF_INCLUDE_DIRS})