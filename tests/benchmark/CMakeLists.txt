if(GLOW_WITH_CPU)
add_executable(GemmBench
               GemmBench.cpp)
target_link_libraries(GemmBench
                      PRIVATE
                        CPURuntimeNative)
endif()
