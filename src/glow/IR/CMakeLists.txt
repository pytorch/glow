
add_library(IR
              IR.cpp
              IRGen.cpp
              IRBuilder.cpp
              Instrs.cpp)
target_link_libraries(IR
                      PUBLIC
                        Base
                        Support)

