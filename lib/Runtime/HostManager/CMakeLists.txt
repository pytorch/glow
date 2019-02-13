add_library(HostManager
              HostManager.cpp)

target_link_libraries(HostManager
                      PRIVATE
                        Backends
                        Base
                        Graph
                        Partitioner
                        Provisioner
                        Executor
                        DeviceManager
                        CPUDeviceManager)
