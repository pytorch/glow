add_library(HostManager
              HostManager.cpp)

target_link_libraries(HostManager
                      PRIVATE
                        Backends
                        Base
                        Executor
                        Graph
                        GraphOptimizer
                        Partitioner
                        Provisioner
                        Runtime)
