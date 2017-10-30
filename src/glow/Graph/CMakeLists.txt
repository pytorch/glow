
set(HDR ${CMAKE_CURRENT_BINARY_DIR}/AutoGenNodes.h)
set(SRC ${CMAKE_CURRENT_BINARY_DIR}/AutoGenNodes.cpp)
set(DEF ${CMAKE_CURRENT_BINARY_DIR}/AutoGenNodes.def)

add_custom_command(OUTPUT
                    "${HDR}"
                    "${SRC}"
                    "${DEF}"
                   COMMAND
                     NodeGen ${HDR} ${SRC} ${DEF}
                   DEPENDS
                     NodeGen
                   COMMENT
                     "NodeGen: Generating nodes."
                   VERBATIM)
add_custom_target(AutoGenGraph
                  DEPENDS
                    ${HDR}
                    ${SRC}
                    ${DEF})

add_library(Graph
              ${CMAKE_SOURCE_DIR}/include/glow/Graph/Graph.h
              ${CMAKE_SOURCE_DIR}/include/glow/Graph/Node.h
              ${CMAKE_SOURCE_DIR}/include/glow/Graph/Nodes.h

              ${CMAKE_SOURCE_DIR}/src/glow/Graph/Graph.cpp
              ${CMAKE_SOURCE_DIR}/src/glow/Graph/Nodes.cpp

              ${HDR}
              ${SRC}
              ${DEF})
target_include_directories(Graph
                           PUBLIC
                             ${CMAKE_CURRENT_BINARY_DIR}
                             $<TARGET_PROPERTY:IR,INCLUDE_DIRECTORIES>)
target_link_libraries(Graph
                      PUBLIC
                        Support)
add_dependencies(Graph
                   AutoGenGraph
                   AutoGenIR)

