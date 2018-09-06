include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_library(onnxifi-glow
            SHARED
              Base.cpp
              onnxifiGlow.cpp)

target_link_libraries(onnxifi-glow
                      PUBLIC
                        ExecutionEngine
                        Importer)
