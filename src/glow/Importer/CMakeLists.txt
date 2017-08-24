find_package(Protobuf REQUIRED)

include_directories(${PROTOBUF_INCLUDE_DIRS})

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS caffe.proto)

add_library (Importer
             ./Caffe2.cpp
             ${PROTO_SRCS}
             ${PROTO_HDRS})

SET(LinkLibs ${LinkLibs} ${PROTOBUF_LIBRARY} PARENT_SCOPE)
