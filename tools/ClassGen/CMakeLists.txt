add_executable(NodeGen
                 NodeGen.cpp
                 NodeBuilder.cpp)

add_executable(InstrGen
                 InstrGen.cpp
                 InstrBuilder.cpp)

target_link_libraries(NodeGen
                      PUBLIC
                        MemberType
                        Support)

target_link_libraries(InstrGen
                      PUBLIC
                        MemberType
                        Support)

add_library(MemberType
              MemberType.cpp)

if(GLOW_WITH_CPU)
  add_subdirectory(Backends/CPU)
endif()

if(GLOW_WITH_OPENCL)
  add_subdirectory(Backends/OpenCL)
endif()

if(GLOW_WITH_NNPI)
  add_subdirectory(Backends/NNPI)
endif()

if(GLOW_WITH_HABANA)
  add_subdirectory(Backends/Habana)
endif()

# External backends
ExternalBackendsClassGen()
