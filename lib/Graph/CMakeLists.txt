# AutoGenInstr
set(INSTR_HDR ${GLOW_BINARY_DIR}/AutoGenInstr.h)
set(INSTR_SRC ${GLOW_BINARY_DIR}/AutoGenInstr.cpp)
set(INSTR_DEF ${GLOW_BINARY_DIR}/AutoGenInstr.def)
set(INSTR_BLD_HDR ${GLOW_BINARY_DIR}/AutoGenIRBuilder.h)
set(INSTR_BLD_SRC ${GLOW_BINARY_DIR}/AutoGenIRBuilder.cpp)
set(INSTR_IR_GEN ${GLOW_BINARY_DIR}/AutoGenIRGen.h)

add_custom_command(OUTPUT
                    "${INSTR_HDR}"
                    "${INSTR_SRC}"
                    "${INSTR_DEF}"
                    "${INSTR_BLD_HDR}"
                    "${INSTR_BLD_SRC}"
                    "${INSTR_IR_GEN}"
                    COMMAND InstrGen ${INSTR_HDR} ${INSTR_SRC} ${INSTR_DEF} ${INSTR_BLD_HDR} ${INSTR_BLD_SRC} ${INSTR_IR_GEN}
                    DEPENDS InstrGen
                    COMMENT "InstrGen: Generating instructions." VERBATIM)

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

add_library(Graph
            ${NODES_HDR}
            ${NODES_SRC}
            ${NODES_DEF}
            ${NODES_BLD}
            ${INSTR_HDR}
            ${INSTR_SRC}
            ${INSTR_DEF}
            ${INSTR_BLD_HDR}
            ${INSTR_BLD_SRC}
            Node.cpp
            Nodes.cpp
            Graph.cpp
            Grad.cpp)

target_link_libraries(Graph
                      PUBLIC
                        Base
                        Support)
