
add_library(Graph
            Nodes.cpp
            Graph.cpp)
target_link_libraries(Graph
                      PUBLIC
                        IR
                        Support)

