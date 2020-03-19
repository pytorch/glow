if (GLOW_BUILD_TESTS)
add_executable(network-debugger
               network-debugger.cpp
               NetworkComparator.cpp)

target_link_libraries(network-debugger
                      PRIVATE
                        BackendTestUtils
                        Importer
                        HostManager
                        ExecutionEngine)
endif()
