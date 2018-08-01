add_library(IR
              IR.cpp
              IRGen.cpp
              IRUtils.cpp
              IRBuilder.cpp
              Instrs.cpp
              GraphScheduler.cpp)

target_link_libraries(IR
                      PUBLIC
                        Graph
                        Base
                        Support)
