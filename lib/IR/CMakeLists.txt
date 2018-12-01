# AutoGenInstr
set(INSTR_HDR ${GLOW_BINARY_DIR}/glow/AutoGenInstr.h)
set(INSTR_SRC ${GLOW_BINARY_DIR}/glow/AutoGenInstr.cpp)
set(INSTR_DEF ${GLOW_BINARY_DIR}/glow/AutoGenInstr.def)
set(INSTR_BLD_HDR ${GLOW_BINARY_DIR}/glow/AutoGenIRBuilder.h)
set(INSTR_BLD_SRC ${GLOW_BINARY_DIR}/glow/AutoGenIRBuilder.cpp)
set(INSTR_IR_GEN ${GLOW_BINARY_DIR}/glow/AutoGenIRGen.h)

add_custom_command(OUTPUT
                    "${INSTR_HDR}"
                    "${INSTR_SRC}"
                    "${INSTR_DEF}"
                    "${INSTR_BLD_HDR}"
                    "${INSTR_BLD_SRC}"
                    "${INSTR_IR_GEN}"
                    COMMAND InstrGen
                      "${INSTR_HDR}" "${INSTR_SRC}" "${INSTR_DEF}"
                      "${INSTR_BLD_HDR}" "${INSTR_BLD_SRC}" "${INSTR_IR_GEN}"
                    DEPENDS InstrGen
                    COMMENT "InstrGen: Generating instructions." VERBATIM)

add_custom_target(IRGenHeaders
                  DEPENDS
                    "${INSTR_HDR}"
                    "${INSTR_DEF}"
                    "${INSTR_BLD_HDR}")

add_library(IR
              "${INSTR_SRC}"
              "${INSTR_BLD_SRC}"
              IR.cpp
              IRGen.cpp
              IRUtils.cpp
              IRBuilder.cpp
              Instrs.cpp
              GraphScheduler.cpp
              ChildMemSizeBasedScheduler.cpp
              TopologicalSortBasedScheduler.cpp)

target_link_libraries(IR
                      PUBLIC
                        Graph
                        Base
                        Support)
