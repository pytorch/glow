add_library(HostManager
              HostManager.cpp)

target_link_libraries(HostManager
                      PRIVATE
                        Backends
                        Base
                        Executor
                        Exporter
                        Graph
                        GraphOptimizer
                        Partitioner
                        Provisioner
                        Runtime)
