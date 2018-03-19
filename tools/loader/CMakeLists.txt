
add_executable(loader
                 loader.cpp)
target_link_libraries(loader
                      PRIVATE
                        Base
                        Importer
                        ExecutionEngine
                        IR
                        Quantization)
