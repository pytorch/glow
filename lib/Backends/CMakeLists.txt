add_subdirectory(Interpreter)

LIST(APPEND linked_device_managers InterpreterDeviceManager)

if(GLOW_WITH_OPENCL)
  add_subdirectory(OpenCL)
  LIST(APPEND linked_backends OpenCL)
  LIST(APPEND linked_device_managers OpenCLDeviceManager)
endif()


if(GLOW_WITH_CPU)
  add_subdirectory(CPU)
  LIST(APPEND linked_backends CPUBackend)
  LIST(APPEND linked_device_managers CPUDeviceManager)
endif()

add_library(Backend
              Backend.cpp
              BackendUtils.cpp
              CompiledFunction.cpp
              TraceEvents.cpp)
target_link_libraries(Backend
                      PRIVATE
                        Base
                        CodeGen
                        Graph
                        IR
                        Optimizer)

add_library(Backends
              Backends.cpp)
target_link_libraries(Backends
                      PRIVATE
                        Base
                        Graph
                        Interpreter
                        Optimizer
                        ${linked_backends})

add_library(DeviceManager
              DeviceManagers.cpp)
target_link_libraries(DeviceManager
                      PUBLIC
                        ThreadPool
                        ${linked_device_managers}
                      PRIVATE
                        Backend  
                        Backends
                        Graph
                        ThreadPool)
