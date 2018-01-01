
set(HDR ${GLOW_BINARY_DIR}/AutoGenInstr.h)
set(SRC ${GLOW_BINARY_DIR}/AutoGenInstr.cpp)
set(DEF ${GLOW_BINARY_DIR}/AutoGenInstr.def)
set(BLD ${GLOW_BINARY_DIR}/AutoGenIRBuilder.h)

add_custom_command(OUTPUT
                    "${HDR}"
                    "${SRC}"
                    "${DEF}"
                    "${BLD}"
                   COMMAND InstrGen ${HDR} ${SRC} ${DEF} ${BLD}
                   DEPENDS InstrGen
                   COMMENT "InstrGen: Generating instrs." VERBATIM)

add_library(Base
              Tensor.cpp
              Type.cpp
              Image.cpp
              ${HDR}
              ${SRC}
              ${DEF}
              ${BLD})

target_link_libraries(Base
                      PUBLIC
                       Support)
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
