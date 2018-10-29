add_library(IR
              IR.cpp
              IRGen.cpp
              IRUtils.cpp
              IRBuilder.cpp
              Instrs.cpp
              GraphScheduler.cpp
              ChildMemSizeBasedScheduler.cpp
              TopologicalSortBasedScheduler.cpp)

target_link_libraries(IR
                      PUBLIC
                        Graph
                        Base
                        Support)
