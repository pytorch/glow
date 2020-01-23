add_executable(NodeGen
                 NodeGen.cpp
                 NodeBuilder.cpp)

add_executable(InstrGen
                 InstrGen.cpp
                 InstrBuilder.cpp)

target_link_libraries(NodeGen
                      PUBLIC
                        MemberType
                        Support)

target_link_libraries(InstrGen
                      PUBLIC
                        MemberType
                        Support)

add_library(MemberType
              MemberType.cpp)

# Include custom nodes and instructions definitions from enabled backends.
set(ClassGen_Include_DIR ${GLOW_BINARY_DIR}/glow)
include_directories(${ClassGen_Include_DIR})

file(REMOVE "${ClassGen_Include_DIR}/NodeGenIncludes.h")
file(REMOVE "${ClassGen_Include_DIR}/InstrGenIncludes.h")

# External backends
ExternalBackendsClassGen()

# Iterate over all enabled backends.
foreach(backend ${GLOW_BACKENDS})
    message(STATUS "Check backend ${backend} for custom instructions or nodes")
    set(backend_classgen_DIR "${GLOW_BACKENDS_DIR}/${backend}/ClassGen")
    # Check for each enabled backend if it has ClassGen subdirectory.
    if(EXISTS ${GLOW_BACKENDS_DIR}/${backend}/ClassGen)
       # Check for header files with custom node definitions in this subdirectory.
       file(GLOB backend_specific_nodes
            RELATIVE "${backend_classgen_DIR}"
            "${backend_classgen_DIR}/*SpecificNodes.h")
       # Include these header files into NodeGenIncludes.h.
       foreach(include_file ${backend_specific_nodes})
           file(APPEND "${ClassGen_Include_DIR}/NodeGenIncludes.h"
                       "#include \"lib/Backends/${backend}/ClassGen/${include_file}\"\n")
       endforeach()
       # Check for header files with custom instruction definitions in this subdirectory.
       file(GLOB backend_specific_instrs
            RELATIVE "${backend_classgen_DIR}"
            "${backend_classgen_DIR}/*SpecificInstrs.h")
       # Include these header files into InstrGenIncludes.h.
       foreach(include_file ${backend_specific_instrs})
           file(APPEND "${ClassGen_Include_DIR}/InstrGenIncludes.h"
                       "#include \"lib/Backends/${backend}/ClassGen/${include_file}\"\n")
       endforeach()
    endif()
endforeach()
