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

add_library(Backends
              Backends.cpp
              DeviceManagers.cpp)

target_link_libraries(Backends
                      PUBLIC
                        ExecutionContext
                      PRIVATE
                        Backend
                        Base
                        Graph)

FOREACH(backend ${linked_backends})
  make_whole_archive(Backends ${backend})
  target_link_libraries(Backends PRIVATE ${backend})
ENDFOREACH()
