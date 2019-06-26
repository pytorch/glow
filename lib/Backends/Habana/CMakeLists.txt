add_library(Habana
            Habana.cpp
            HabanaDeviceManager.cpp
            HabanaFactory.cpp
            HabanaUtils.cpp)

target_include_directories(Habana
                           PUBLIC
                             "${SYNAPSE_INCLUDE_DIR}")

target_link_libraries(Habana
                      PUBLIC
                        Synapse
                      PRIVATE
                        Backend
                        Base
                        Graph
                        CodeGen
                        IR
                        LLVMCore
                        QuantizationBase
                        Support)

set(linked_backends ${linked_backends} Habana PARENT_SCOPE)
