
add_executable(loader
                 loader.cpp)
target_link_libraries(loader
                      PRIVATE
                        Interpreter
                        Importer
                        Network
                        IR
                        Support)

