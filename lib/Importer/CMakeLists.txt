include_directories(${PROTOBUF_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_definitions(-DGOOGLE_PROTOBUF_NO_RTTI)

if(NOT TARGET onnx_proto)
  # Note: This avoids libprotobuf.so complaining about name collisions at runtime
  if(NOT ONNX_NAMESPACE)
    set(ONNX_NAMESPACE "glow_onnx")
  endif()
  add_definitions("-DONNX_NAMESPACE=${ONNX_NAMESPACE}")
  add_subdirectory(${GLOW_THIRDPARTY_DIR}/onnx EXCLUDE_FROM_ALL build_onnx)
endif()

find_package(Protobuf REQUIRED)
PROTOBUF_GENERATE_CPP(CAFFE_SRCS CAFFE_HDRS caffe2.proto)

# NB: We need to copy the *.pb.h files to appropriately-prefixed paths to
# placate FB's internal build system.  That is, we need:
#
#   #include "caffe2/proto/caffe2.pb.h"
#
# instead of:
#
#   #include "caffe2.pb.h"
#
# Please don't optimize away the "useless" copies!
add_custom_command(
  OUTPUT ${GLOW_BINARY_DIR}/caffe2/proto/caffe2.pb.h
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CAFFE_HDRS} ${GLOW_BINARY_DIR}/caffe2/proto/caffe2.pb.h
  DEPENDS ${CAFFE_HDRS})

add_library(Importer
              ProtobufLoader.cpp
              Caffe2ModelLoader.cpp
              ONNXModelLoader.cpp
              ONNXIFIModelLoader.cpp
              ${CAFFE_SRCS}
              ${GLOW_BINARY_DIR}/caffe2/proto/caffe2.pb.h)
target_include_directories(Importer PUBLIC ${ONNX_INCLUDE_DIRS} ${GLOW_THIRDPARTY_DIR}/foxi)
target_compile_definitions(Importer
                           INTERFACE
                             -DGOOGLE_PROTOBUF_NO_RTTI)
target_link_libraries(Importer
                      PRIVATE
                        Backends
                        Base
                        Graph
                        GraphOptimizer
                        LLVMSupport
                        Support)
target_link_libraries(Importer PUBLIC onnx_proto ${PROTOBUF_LIBRARY})

if (MSVC AND LINK_PROTOBUF_AS_DLL)
  # For protobuf warning when it is build as dll.
  # Supresses a warning that is treated as error.
  # Basically one of the header files has interface class
  # containing STL string. Which might cause issues
  # if things are build with different compilers.
  #
  # Sets general warning level as 2 for this project.
  # There are few warnings that are treated as errors that
  # come from VS include headers
  target_compile_options(onnx_proto PUBLIC /wd4251)
  target_compile_options(onnx_proto PUBLIC /W2)
  target_compile_definitions(onnx_proto PUBLIC -DPROTOBUF_USE_DLLS)
endif()
