include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${GLOW_THIRDPARTY_DIR}/onnx)

add_library(onnxifi-glow
            SHARED
              onnxifiGlow.cpp)

target_link_libraries(onnxifi-glow
                      PRIVATE
                        Importer)
