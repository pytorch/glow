add_library(Habana
            Habana.cpp
            SynapseShim.cpp)

target_include_directories(Habana
                           PUBLIC
                             "${SYNAPSE_INCLUDE_DIR}")

target_link_libraries(Habana
                      PRIVATE
                        Backend
                        Base
                        Graph
                        CodeGen
                        IR
                        Optimizer
                        QuantizationBase
                        Support)

add_library(HabanaDeviceManager
            HabanaDeviceManager.cpp)
target_link_libraries(HabanaDeviceManager
                      PRIVATE
                        Backend
                        Habana
                        Base
                        Graph
                        IR
                        Optimizer
                        ThreadPool)

add_library(HabanaFactory
            HabanaFactory.cpp)
target_link_libraries(HabanaFactory
                      PRIVATE
                        Habana
                        LLVMCore)

set(linked_factories ${linked_factories} HabanaFactory PARENT_SCOPE)
set(linked_device_managers ${linked_device_managers} HabanaDeviceManager PARENT_SCOPE)
