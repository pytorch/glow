
set(HDR ${GLOW_BINARY_DIR}/AutoGenNodes.h)
set(SRC ${GLOW_BINARY_DIR}/AutoGenNodes.cpp)
set(DEF ${GLOW_BINARY_DIR}/AutoGenNodes.def)

add_custom_command(OUTPUT
                    "${HDR}"
                    "${SRC}"
                    "${DEF}"
                   COMMAND NodeGen ${HDR} ${SRC} ${DEF}
                   DEPENDS NodeGen
                   COMMENT "NodeGen: Generating nodes." VERBATIM)

add_library(Graph
            Nodes.cpp
            Graph.cpp
            ${HDR}
            ${SRC}
            ${DEF}
            )


target_link_libraries(Graph
                      PUBLIC
                        Base
                        Support)
