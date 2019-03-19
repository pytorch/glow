# Iterate all subdirectories and check whether
# particular backend is enabled.
FILE(GLOB subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(object ${subdirs})
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${object})
    set(backendEnabled GLOW_WITH_${object})
    string(TOUPPER ${backendEnabled} backendEnabled)
    if(NOT DEFINED ${backendEnabled} OR ${backendEnabled})
      message("Adding ${object} backend.")
      add_subdirectory(${object})
    endif()
  endif()
endforeach()

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
