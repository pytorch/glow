add_library(Habana
            Habana.cpp
            HabanaDeviceManager.cpp
            HabanaFactory.cpp)

target_include_directories(Habana
                           PUBLIC
                             "${SYNAPSE_INCLUDE_DIR}")

target_link_libraries(Habana
                      PUBLIC
                        "${SYNAPSE_LIB}"
                      PRIVATE
                        Backend
                        Base
                        Graph
                        CodeGen
                        IR
                        LLVMCore
                        Optimizer
                        QuantizationBase
                        Support)

set(linked_factories ${linked_factories} Habana PARENT_SCOPE)
set(linked_device_managers ${linked_device_managers} Habana PARENT_SCOPE)
