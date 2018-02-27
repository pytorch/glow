# AutoGenBackend [Nodes/Instrs] hooks
set(BACKEND_NODES_HOOK ${GLOW_BINARY_DIR}/AutoGenBackendNodesHook.cpp)
set(BACKEND_INSTRS_HOOK ${GLOW_BINARY_DIR}/AutoGenBackendInstrsHook.cpp)
add_custom_command(OUTPUT
                     "${BACKEND_NODES_HOOK}"
                     "${BACKEND_INSTRS_HOOK}"
                     COMMAND BackendGen ${BACKEND_NODES_HOOK} ${BACKEND_INSTRS_HOOK}
                     DEPENDS BackendGen
                     COMMENT "BackendGen: Generating hooks for backend nodes and instrs." VERBATIM)

if(GLOW_WITH_CPU)
  LIST(APPEND BACKEND_NODES backends/CPUBackend/Nodes.cpp)
  LIST(APPEND BACKEND_INSTRS backends/CPUBackend/Instrs.cpp)
endif()
if(GLOW_WITH_OPENCL)
  LIST(APPEND BACKEND_NODES backends/OpenCL/Nodes.cpp)
  LIST(APPEND BACKEND_INSTRS backends/OpenCL/Instrs.cpp)
endif()

add_executable(BackendGen
                 BackendGen.cpp)

add_executable(NodeGen
                 NodeGen.cpp
                 NodeBuilder.cpp
                 ${BACKEND_NODES_HOOK}
                 ${BACKEND_NODES})

add_executable(InstrGen
                 InstrGen.cpp
                 InstrBuilder.cpp
                 ${BACKEND_INSTRS_HOOK}
                 ${BACKEND_INSTRS})

if(GLOW_WITH_CPU)
  target_compile_definitions(NodeGen PRIVATE GLOW_WITH_CPU=1)
  target_compile_definitions(InstrGen PRIVATE GLOW_WITH_CPU=1)
endif()
if(GLOW_WITH_OPENCL)
  target_compile_definitions(NodeGen PRIVATE GLOW_WITH_OPENCL=1)
  target_compile_definitions(InstrGen PRIVATE GLOW_WITH_OPENCL=1)
endif()

target_link_libraries(NodeGen
                      PUBLIC
                        Support)

target_link_libraries(InstrGen
                      PUBLIC
                        Support)
