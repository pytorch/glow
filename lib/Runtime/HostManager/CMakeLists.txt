add_library(HostManager
              HostManager.cpp)

target_link_libraries(HostManager
                      PRIVATE
                        Backends
                        Base
                        Graph
                        GraphOptimizer
                        Partitioner
                        Provisioner
                        Executor)
