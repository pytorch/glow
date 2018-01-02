add_library(Graph
            Nodes.cpp
            Graph.cpp
            Grad.cpp)


target_link_libraries(Graph
                      PUBLIC
                        Base
                        Support)
