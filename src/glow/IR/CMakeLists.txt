
set(HDR ${CMAKE_CURRENT_BINARY_DIR}/AutoGenInstr.h)
set(SRC ${CMAKE_CURRENT_BINARY_DIR}/AutoGenInstr.cpp)
set(DEF ${CMAKE_CURRENT_BINARY_DIR}/AutoGenInstr.def)

add_custom_command(OUTPUT
                    "${HDR}"
                    "${SRC}"
                    "${DEF}"
                   COMMAND
                     InstrGen ${HDR} ${SRC} ${DEF}
                   DEPENDS
                     InstrGen
                   COMMENT
                     "InstrGen: Generating instrs."
                   VERBATIM)
add_custom_target(AutoGenIR
                  DEPENDS
                    ${HDR}
                    ${SRC}
                    ${DEF})


add_library(IR
              ${CMAKE_SOURCE_DIR}/include/glow/IR/IR.h
              ${CMAKE_SOURCE_DIR}/include/glow/IR/IRBuilder.h
              ${CMAKE_SOURCE_DIR}/include/glow/IR/Instrs.h
              ${CMAKE_SOURCE_DIR}/include/glow/IR/UseDef.h

              ${CMAKE_SOURCE_DIR}/src/glow/IR/IR.cpp
              ${CMAKE_SOURCE_DIR}/src/glow/IR/IRGen.cpp
              ${CMAKE_SOURCE_DIR}/src/glow/IR/IRBuilder.cpp
              ${CMAKE_SOURCE_DIR}/src/glow/IR/Instrs.cpp

              ${HDR}
              ${SRC}
              ${DEF})
target_include_directories(IR
                           PUBLIC
                             ${CMAKE_CURRENT_BINARY_DIR}
                             $<TARGET_PROPERTY:Graph,INCLUDE_DIRECTORIES>)
target_link_libraries(IR
                      INTERFACE
                        Graph)
add_dependencies(IR
                   AutoGenGraph
                   AutoGenIR)

