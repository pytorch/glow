add_library(Partitioner
            Partitioner.cpp)

target_link_libraries(Partitioner
                      PRIVATE
                        Graph)
