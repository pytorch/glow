add_subdirectory(Interpreter)

if(GLOW_WITH_OPENCL)
  add_subdirectory(OpenCL)
  LIST(APPEND linked_backends OpenCL)
endif()


if(GLOW_WITH_CPU)
  add_subdirectory(CPU)
  LIST(APPEND linked_backends CPUBackend)
  LIST(APPEND linked_device_managers CPUDeviceManager)
endif()

add_library(Backends
              Backend.cpp
              Backends.cpp)
target_link_libraries(Backends
                      PRIVATE
                        BackendUtils
                        Base
                        Graph
                        Interpreter
                        Optimizer
                        ${linked_backends})

add_library(BackendUtils BackendUtils.cpp)
target_link_libraries(BackendUtils
                      PRIVATE
                      CodeGen
                      IR)

add_library(DeviceManager
              DeviceManagers.cpp
              QueueBackedDeviceManager.cpp)
target_link_libraries(DeviceManager
                      PRIVATE
                        Backends
                        Graph
                        ThreadPool
                        ${linked_device_managers})

add_library(QueueBackedDeviceManager
            QueueBackedDeviceManager.cpp)
target_link_libraries(QueueBackedDeviceManager
                      PRIVATE
                        Backends
                        Graph
                        ThreadPool)
