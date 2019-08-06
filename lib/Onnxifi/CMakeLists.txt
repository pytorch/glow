include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_library(onnxifi-glow
            SHARED
              Base.cpp
              GlowOnnxifiManager.cpp
              HostManagerOnnxifi.cpp
              onnxifiGlow.cpp
              InlineOnnxifi.cpp)

add_library(onnxifi-glow-lib
              Base.cpp
              GlowOnnxifiManager.cpp
              HostManagerOnnxifi.cpp
              onnxifiGlow.cpp
              InlineOnnxifi.cpp)

target_link_libraries(onnxifi-glow-lib
                      PUBLIC
                        Backends
                        ExecutionContext
                        ExecutionEngine
                        Graph
                        HostManager
                        Importer
                        GraphOptimizer
                        Quantization)

target_link_libraries(onnxifi-glow
                      PUBLIC
                        Backends
                        ExecutionContext
                        ExecutionEngine
                        Graph
                        HostManager
                        Importer
                        GraphOptimizer
                        Quantization)

target_compile_definitions(onnxifi-glow
                           PRIVATE
                             ONNXIFI_BUILD_LIBRARY)

target_compile_definitions(onnxifi-glow-lib
                           PRIVATE
                             ONNXIFI_BUILD_LIBRARY)
