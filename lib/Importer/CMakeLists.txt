include_directories(${PROTOBUF_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_definitions(-DGOOGLE_PROTOBUF_NO_RTTI)

if(NOT TARGET onnx_proto)
  # Note: This avoids libprotobuf.so complaining about name collisions at runtime
  if(NOT ONNX_NAMESPACE)
    set(ONNX_NAMESPACE "glow_onnx")
  endif()
  add_definitions("-DONNX_NAMESPACE=${ONNX_NAMESPACE}")
  add_subdirectory(${GLOW_THIRDPARTY_DIR}/onnx EXCLUDE_FROM_ALL build)
endif()

find_package(Protobuf REQUIRED)
PROTOBUF_GENERATE_CPP(CAFFE_SRCS CAFFE_HDRS caffe.proto)

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
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CAFFE_HDRS} ${GLOW_BINARY_DIR}/glow/caffe.pb.h
  DEPENDS ${CAFFE_HDRS})

add_library(Importer
              ProtobufLoader.cpp
              Caffe2.cpp
              ONNX.cpp
              ONNXIFILoader.cpp 
              ${CAFFE_SRCS}
              ${GLOW_BINARY_DIR}/glow/caffe.pb.h)
target_include_directories(Importer PUBLIC ${ONNX_INCLUDE_DIRS})
target_compile_definitions(Importer
                           INTERFACE
                             -DGOOGLE_PROTOBUF_NO_RTTI)
target_link_libraries(Importer
                      PRIVATE
                        Base
                        Graph
                        Support)
target_link_libraries(Importer PUBLIC onnx_proto ${PROTOBUF_LIBRARY})

