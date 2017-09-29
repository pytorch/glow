
add_library(IR
              IR.cpp
              Type.cpp
              IRBuilder.cpp
              Instrs.cpp)
target_link_libraries(IR
                      PUBLIC
                        Support)

