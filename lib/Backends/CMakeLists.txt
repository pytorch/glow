add_library(Backends
              Backend.cpp
              Backends.cpp)

add_library(BackendUtils BackendUtils.cpp)

target_link_libraries(BackendUtils
                      PRIVATE
                      CodeGen
                      IR)

add_subdirectory(Interpreter)

if(GLOW_WITH_OPENCL)
  add_subdirectory(OpenCL)
  LIST(APPEND linked_backends OpenCL)
endif()


if(GLOW_WITH_CPU)
  add_subdirectory(CPU)
  LIST(APPEND linked_backends CPUBackend)
endif()

target_link_libraries(Backends
                      PRIVATE
                        BackendUtils
                        Interpreter
                        ${linked_backends}
                        Base
                        Graph)

add_library(DeviceManager QueueBackedDeviceManager.cpp)

target_link_libraries(DeviceManager
                      PRIVATE
                        Backends
                        Graph
                        ThreadPool)
