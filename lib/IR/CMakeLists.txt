
add_library(IR
              IR.cpp
              IRGen.cpp
              IRBuilder.cpp
              Instrs.cpp
              GraphScheduler.cpp)

target_link_libraries(IR
                      PUBLIC
                        Graph
                        Base
                        Support)
