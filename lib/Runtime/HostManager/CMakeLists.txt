add_library(HostManager
              HostManager.cpp)

target_link_libraries(HostManager
                      PRIVATE
                        Backend
                        Backends
                        Base
                        Graph
                        Partitioner
                        Provisioner
                        Executor
                        DeviceManager
                        CPUDeviceManager)
