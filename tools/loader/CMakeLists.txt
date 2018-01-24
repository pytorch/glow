
add_executable(loader
                 loader.cpp)
target_link_libraries(loader
                      PRIVATE
                        Backends
                        Importer
                        ExecutionEngine
                        IR
                        Support
                        Quantization)

