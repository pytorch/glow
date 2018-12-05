add_library(HostManager
              HostManager.cpp)

target_link_libraries(HostManager
                      PRIVATE
                        Backends
                        Provisioner
                        Executor
                        DeviceManager
                        CPUDeviceManager)                       
