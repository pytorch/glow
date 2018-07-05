find_package(Protobuf REQUIRED)

include_directories(${PROTOBUF_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_definitions(-DGOOGLE_PROTOBUF_NO_RTTI)

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS caffe.proto onnx.proto)

add_library(Importer
              ProtobufLoader.cpp
              Caffe2.cpp
              ONNX.cpp
              ${PROTO_SRCS}
              ${PROTO_HDRS})
target_compile_definitions(Importer
                           INTERFACE
                             -DGOOGLE_PROTOBUF_NO_RTTI)
target_link_libraries(Importer
                      PRIVATE
                        Base
                        Graph
                        Support
                        ${PROTOBUF_LIBRARY})
