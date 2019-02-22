add_subdirectory(Interpreter)

LIST(APPEND linked_device_managers InterpreterDeviceManager)
LIST(APPEND linked_factories InterpreterFactory)

if(GLOW_WITH_OPENCL)
  add_subdirectory(OpenCL)
  LIST(APPEND linked_factories OpenCLFactory)
  LIST(APPEND linked_device_managers OpenCLDeviceManager)
endif()


if(GLOW_WITH_CPU)
  add_subdirectory(CPU)
  LIST(APPEND linked_factories CPUFactory)
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
                        Optimizer)

FOREACH(factory ${linked_factories})
  make_whole_archive(Backends ${factory})
  target_link_libraries(Backends PRIVATE ${factory})
ENDFOREACH()

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
