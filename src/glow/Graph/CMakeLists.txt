
add_library(Graph
            IRGen.cpp
            Nodes.cpp
            Graph.cpp)
target_link_libraries(Graph
                      PUBLIC
                        IR
                        Support)

