
# AutoGenInstr
set(INSTR_HDR ${GLOW_BINARY_DIR}/AutoGenInstr.h)
set(INSTR_SRC ${GLOW_BINARY_DIR}/AutoGenInstr.cpp)
set(INSTR_DEF ${GLOW_BINARY_DIR}/AutoGenInstr.def)
set(INSTR_BLD ${GLOW_BINARY_DIR}/AutoGenIRBuilder.h)

add_custom_command(OUTPUT
                    "${INSTR_HDR}"
                    "${INSTR_SRC}"
                    "${INSTR_DEF}"
                    "${INSTR_BLD}"
                    COMMAND InstrGen ${INSTR_HDR} ${INSTR_SRC} ${INSTR_DEF} ${INSTR_BLD}
                    DEPENDS InstrGen
                    COMMENT "InstrGen: Generating instructions." VERBATIM)

add_library(AutoGenInstr
            ${INSTR_HDR}
            ${INSTR_SRC}
            ${INSTR_DEF}
            ${INSTR_BLD})

target_link_libraries(AutoGenInstr PUBLIC Support)

# AutoGenNodes
set(NODES_HDR ${GLOW_BINARY_DIR}/AutoGenNodes.h)
set(NODES_SRC ${GLOW_BINARY_DIR}/AutoGenNodes.cpp)
set(NODES_DEF ${GLOW_BINARY_DIR}/AutoGenNodes.def)

add_custom_command(OUTPUT
                   "${NODES_HDR}"
                   "${NODES_SRC}"
                   "${NODES_DEF}"
                   COMMAND NodeGen ${NODES_HDR} ${NODES_SRC} ${NODES_DEF}
                   DEPENDS NodeGen
                   COMMENT "NodeGen: Generating nodes." VERBATIM)

add_library(AutoGenNodes
            ${NODES_HDR}
            ${NODES_SRC}
            ${NODES_DEF}
            ${NODES_BLD})

target_link_libraries(AutoGenNodes PUBLIC Support)

# Base
add_library(Base
              Tensor.cpp
              Type.cpp
              Image.cpp)

target_link_libraries(Base
                      PUBLIC
                        Support
                        AutoGenInstr
                        AutoGenNodes)
if(PNG_FOUND)
  target_compile_definitions(Base
                             PRIVATE
                               WITH_PNG=1)
  target_include_directories(Base
                             PRIVATE
                               ${PNG_INCLUDE_DIR})
  target_link_libraries(Base
                        PRIVATE
                          ${PNG_LIBRARY})
endif()
