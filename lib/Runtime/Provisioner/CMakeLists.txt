add_library(Provisioner
              Provisioner.cpp)

target_link_libraries(Provisioner
                      PRIVATE
                        Backend
                        Backends
                        Flags
                        Graph
                        Runtime)
