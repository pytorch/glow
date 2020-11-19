include_directories(${CMAKE_CURRENT_BINARY_DIR})

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
                        Exporter
                        Flags
                        Graph
                        HostManager
                        Importer
                        GraphOptimizer
                        Quantization
                        Runtime)

target_compile_definitions(onnxifi-glow-lib
                           PRIVATE
                             ONNXIFI_BUILD_LIBRARY)

if(GLOW_BUILD_ONNXIFI_DYNLIB)
  add_library(onnxifi-glow
              SHARED
                Base.cpp
                GlowOnnxifiManager.cpp
                HostManagerOnnxifi.cpp
                onnxifiGlow.cpp
                InlineOnnxifi.cpp)
  set_target_properties(onnxifi-glow PROPERTIES OUTPUT_NAME "onnxifi")

  target_link_libraries(onnxifi-glow
                        PUBLIC
                          Backends
                          ExecutionContext
                          ExecutionEngine
                          Exporter
                          Flags
                          Graph
                          HostManager
                          Importer
                          GraphOptimizer
                          Quantization
                          Runtime
                          gflags)

  target_compile_definitions(onnxifi-glow
                             PRIVATE
                               ONNXIFI_BUILD_LIBRARY)
endif()
