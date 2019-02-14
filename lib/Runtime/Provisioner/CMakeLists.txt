add_library(Provisioner
              Provisioner.cpp)

target_link_libraries(Provisioner
                      PRIVATE
                        Backends
                        BackendUtils
                        Graph)
