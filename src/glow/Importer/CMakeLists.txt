find_package(Protobuf REQUIRED)

include_directories(${PROTOBUF_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS caffe.proto)

add_library(Importer
              Caffe2.cpp
               ${PROTO_SRCS}
               ${PROTO_HDRS})
target_link_libraries(Importer
                      PRIVATE
                        IR
                        ${PROTOBUF_LIBRARY})

