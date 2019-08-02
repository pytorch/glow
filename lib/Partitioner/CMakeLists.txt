add_library(Partitioner
              PartitionerUtils.cpp
              PartitionerOptimizer.cpp
              PartitionerValidation.cpp
              Partitioner.cpp)

target_link_libraries(Partitioner
                      PRIVATE
                        Backends
                        Graph
                        GraphOptimizer)
