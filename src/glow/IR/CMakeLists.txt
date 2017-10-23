
set(HDR ${GLOW_BINARY_DIR}/AutoGenInstr.h)
set(SRC ${GLOW_BINARY_DIR}/AutoGenInstr.cpp)
set(DEF ${GLOW_BINARY_DIR}/AutoGenInstr.def)

add_custom_command(OUTPUT
                    "${HDR}"
                    "${SRC}"
                    "${DEF}"
                   COMMAND InstrGen ${HDR} ${SRC} ${DEF}
                   DEPENDS InstrGen
                   COMMENT "InstrGen: Generating instrs." VERBATIM)

# Add a library that's dedicated to the auto-generated IR to break the circular
# dependency on the Graph library that uses the generated def file.
add_library(AutoGenIR
              ${HDR}
              ${SRC}
              ${DEF})
target_link_libraries(AutoGenIR
                      PUBLIC
                        Support)

add_library(IR
              IR.cpp
              IRGen.cpp
              IRBuilder.cpp
              Instrs.cpp)

target_link_libraries(IR
                      PUBLIC
                        AutoGenIR
                        Graph
                        Base
                        Support)

