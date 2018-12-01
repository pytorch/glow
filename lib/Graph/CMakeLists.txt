# AutoGenNodes
set(NODES_HDR ${GLOW_BINARY_DIR}/glow/AutoGenNodes.h)
set(NODES_SRC ${GLOW_BINARY_DIR}/glow/AutoGenNodes.cpp)
set(NODES_DEF ${GLOW_BINARY_DIR}/glow/AutoGenNodes.def)

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
            Context.cpp
            Node.cpp
            Nodes.cpp
            NodeValue.cpp
            Graph.cpp
            Grad.cpp
            VerifierHelper.cpp)

target_link_libraries(Graph
                      PUBLIC
                        Base
                        Support
                        QuantizationBase)

add_dependencies(Graph IRGenHeaders)
