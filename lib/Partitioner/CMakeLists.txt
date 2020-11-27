add_library(Partitioner
              PartitionerBase.cpp
              PartitionerUtils.cpp
              PartitionerOptimizer.cpp
              PartitionerValidation.cpp
              Partitioner.cpp)

target_link_libraries(Partitioner
                      PRIVATE
                        Backends
                        Flags
                        Graph
                        GraphOptimizer)
