# Iterate all subdirectories and check whether particular backend is enabled.
FILE(GLOB subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(object ${subdirs})
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${object})
    set(backendEnabled GLOW_WITH_${object})
    string(TOUPPER ${backendEnabled} backendEnabled)
    # Add sub directory for a backend if this backend is enabled
    # or there is no defined variable for that backend.
    # For example, Interpreter backend will be added automatically.
    # While there is GLOW_WITH_CPU variable to control whether
    # CPU backend is enabled or not.
    if(NOT DEFINED ${backendEnabled} OR ${backendEnabled})
      message("Adding ${object} backend.")
      add_subdirectory(${object})
    endif()
  endif()
endforeach()

# External backends
ExternalBackendsRegister()

add_library(ExecutionContext
              TraceEvents.cpp)
target_link_libraries(ExecutionContext
                      PRIVATE
                        Base
                        Graph)

add_library(Backend
              Backend.cpp
              BackendUtils.cpp
              CompiledFunction.cpp)
target_link_libraries(Backend
                      PUBLIC
                        ExecutionContext
                      PRIVATE
                        Base
                        CodeGen
                        Graph
                        IR)
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
                        ExecutionContext
                        ${linked_device_managers}
                      PRIVATE
                        Backend
                        Backends
                        Graph
                        ThreadPool)
