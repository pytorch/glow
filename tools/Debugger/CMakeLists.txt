add_executable(network-debugger
               network-debugger.cpp
               NetworkComparator.cpp)

target_link_libraries(network-debugger
                      PRIVATE
                        BackendTestUtils
                        HostManager
                        Importer
                        HostManager
                        ExecutionEngine
                        Exporter)
