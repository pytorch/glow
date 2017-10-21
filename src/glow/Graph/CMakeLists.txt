
set(HDR ${GLOW_BINARY_DIR}/AutoGenNodes.h)
set(SRC ${GLOW_BINARY_DIR}/AutoGenNodes.cpp)

add_custom_command(OUTPUT
                    "${HDR}"
                    "${SRC}"
                   COMMAND NodeGen ${HDR} ${SRC}
                   DEPENDS NodeGen
                   COMMENT "NodeGen: Generating nodes." VERBATIM)

add_library(Graph
            Nodes.cpp
            Graph.cpp
            ${HDR}
            ${SRC}
            )


target_link_libraries(Graph
                      PUBLIC
                        Base
                        Support)
