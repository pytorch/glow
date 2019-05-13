add_library(HostManager
              HostManager.cpp)

target_link_libraries(HostManager
                      PRIVATE
                        Backends
                        Base
                        Graph
                        Optimizer
                        Partitioner
                        Provisioner
                        Executor)
