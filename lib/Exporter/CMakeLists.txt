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

if (MSVC AND LINK_PROTOBUF_AS_DLL)
  # For protobuf warning when it is build as dll.
  # Suppresses a warning that is treated as error.
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
