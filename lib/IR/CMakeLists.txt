
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

if(GLOW_WITH_CPU)
  target_compile_definitions(IR PRIVATE GLOW_WITH_CPU=1)
endif()
if(GLOW_WITH_OPENCL)
  target_compile_definitions(IR PRIVATE GLOW_WITH_OPENCL=1)
endif()
