# AutoGenNodes
set(NODES_HDR ${GLOW_BINARY_DIR}/glow/AutoGenNodes.h)
set(NODES_SRC ${GLOW_BINARY_DIR}/glow/AutoGenNodes.cpp)
set(NODES_DEF ${GLOW_BINARY_DIR}/glow/AutoGenNodes.def)
set(NODES_IMPORT ${GLOW_BINARY_DIR}/glow/AutoGenNodesImport.h)
set(NODES_EXPORT ${GLOW_BINARY_DIR}/glow/AutoGenNodesExport.h)

add_custom_command(OUTPUT
                   "${NODES_HDR}"
                   "${NODES_SRC}"
                   "${NODES_DEF}"
                   "${NODES_IMPORT}"
                   "${NODES_EXPORT}"
                   COMMAND NodeGen ${NODES_HDR} ${NODES_SRC} ${NODES_DEF} ${NODES_IMPORT} ${NODES_EXPORT}
                   DEPENDS NodeGen
                   COMMENT "NodeGen: Generating nodes." VERBATIM)
add_custom_target(AutoGenNode
                   DEPENDS
                     "${NODES_HDR}"
                     "${NODES_SRC}"
                     "${NODES_DEF}"
                     "${NODES_IMPORT}"
                     "${NODES_EXPORT}")
add_dependencies(AutoGen AutoGenNode)

add_library(Graph
            ${NODES_HDR}
            ${NODES_SRC}
            ${NODES_DEF}
            Hook.cpp
            Node.cpp
            Nodes.cpp
            NodeValue.cpp
            Log.cpp
            PlaceholderBindings.cpp
            TensorLayout.cpp
            Graph.cpp
            GraphUtils.cpp
            Grad.cpp
            VerifierHelper.cpp)

target_link_libraries(Graph
                      PUBLIC
                        Base
                        Flags
                        Support
                        QuantizationBase
                        TensorPool)

add_dependencies(Graph AutoGen)
