include_directories(${PROTOBUF_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_definitions(-DGOOGLE_PROTOBUF_NO_RTTI)

add_library(Exporter
              ProtobufWriter.cpp
              ONNXModelWriter.cpp)
target_compile_definitions(Exporter
                           INTERFACE
                             -DGOOGLE_PROTOBUF_NO_RTTI)
target_link_libraries(Exporter
                      PRIVATE
                        Base
                        Graph
                        Importer
                        LLVMSupport
                        Support)
target_link_libraries(Exporter PUBLIC onnx_proto ${PROTOBUF_LIBRARY})
