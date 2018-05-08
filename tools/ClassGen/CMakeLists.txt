add_executable(NodeGen
                 NodeGen.cpp
                 NodeBuilder.cpp)

add_executable(InstrGen
                 InstrGen.cpp
                 InstrBuilder.cpp)

target_link_libraries(NodeGen
                      PUBLIC
                        Support)

target_link_libraries(InstrGen
                      PUBLIC
                        Support)

if(GLOW_WITH_CPU)
  add_subdirectory(Backends/CPU)
endif()

if(GLOW_WITH_OPENCL)
  add_subdirectory(Backends/OpenCL)
endif()
