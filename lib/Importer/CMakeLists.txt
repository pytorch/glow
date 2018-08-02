find_package(Protobuf REQUIRED)

include_directories(${PROTOBUF_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_definitions(-DGOOGLE_PROTOBUF_NO_RTTI)

PROTOBUF_GENERATE_CPP(CAFFE_SRCS CAFFE_HDRS caffe.proto)
PROTOBUF_GENERATE_CPP(ONNX_SRCS
                      ONNX_HDRS
                      ${GLOW_THIRDPARTY_DIR}/onnx/onnx/onnx.proto)

# NB: We need to copy the *.pb.h files to appropriately-prefixed paths to
# placate FB's internal build system.  That is, we need:
#
#   #include "glow/caffe.pb.h"
#
# instead of:
#
#   #include "caffe.pb.h"
#
# Please don't optimize away the "useless" copies!
add_custom_command(
  OUTPUT ${GLOW_BINARY_DIR}/glow/caffe.pb.h
  COMMAND cp ${CAFFE_HDRS} ${GLOW_BINARY_DIR}/glow/caffe.pb.h
  DEPENDS ${CAFFE_HDRS})

add_custom_command(
  OUTPUT ${GLOW_BINARY_DIR}/onnx/onnx.pb.h
  COMMAND cp ${ONNX_HDRS} ${GLOW_BINARY_DIR}/onnx/onnx.pb.h
  DEPENDS ${ONNX_HDRS})

add_library(Importer
              ProtobufLoader.cpp
              Caffe2.cpp
              ONNX.cpp
              ${CAFFE_SRCS}
              ${GLOW_BINARY_DIR}/glow/caffe.pb.h
              ${ONNX_SRCS}
              ${GLOW_BINARY_DIR}/onnx/onnx.pb.h)
target_compile_definitions(Importer
                           INTERFACE
                             -DGOOGLE_PROTOBUF_NO_RTTI)
target_link_libraries(Importer
                      PRIVATE
                        Base
                        Graph
                        Support
                        ${PROTOBUF_LIBRARY})
