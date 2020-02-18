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

# Include custom ONNX model writers from enabled backends.
set(Exporter_Include_DIR ${GLOW_BINARY_DIR}/glow)
include_directories(${Exporter_Include_DIR})

file(REMOVE "${Exporter_Include_DIR}/ONNXModelWriterIncludes.h")

# External backends
ExternalBackendsCollectONNXModelWriters()

# Iterate over all enabled backends.
foreach(backend ${GLOW_BACKENDS})
    message(STATUS "Check backend ${backend} for ONNXModelWriters")
    set(backend_ONNX_DIR "${GLOW_BACKENDS_DIR}/${backend}/ONNX")
    # Check for ONNXModelWriters in the current backend subdirectory.
    file(GLOB backend_specific_onnx_model_writers
            RELATIVE "${backend_ONNX_DIR}"
            "${backend_ONNX_DIR}/*ONNXModelWriter.cpp")
    # Include these files into ONNXModelWriterIncludes.h.
    foreach(onnx_model_writer ${backend_specific_onnx_model_writers})
           file(APPEND "${Exporter_Include_DIR}/ONNXModelWriterIncludes.h"
                       "#include \"lib/Backends/${backend}/ONNX/${onnx_model_writer}\"\n")
    endforeach()
endforeach()
