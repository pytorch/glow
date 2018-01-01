
add_executable(NodeGen
                 NodeGen.cpp
                 NodeBuilder.cpp)

add_executable(InstrGen
                 InstrGen.cpp
                 InstrBuilder.cpp)

target_link_libraries(NodeGen
                      PUBLIC
                        Support)

target_link_libraries(InstrGen
                      PUBLIC
                        Support)
