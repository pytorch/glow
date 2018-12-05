include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_library(onnxifi-glow
            SHARED
              Base.cpp
              GlowOnnxifiManager.cpp
              onnxifiGlow.cpp)

add_library(onnxifi-glow-normal
              Base.cpp
              GlowOnnxifiManager.cpp
              onnxifiGlow.cpp)

add_library(onnxifi-glow-thread-pool
              ThreadPool.cpp)


target_link_libraries(onnxifi-glow-normal
                      PUBLIC
                        ExecutionEngine
                        Importer
                      PRIVATE
                        onnxifi-glow-thread-pool)

target_link_libraries(onnxifi-glow
                      PUBLIC
                        ExecutionEngine
                        Importer
                      PRIVATE
                        onnxifi-glow-thread-pool)
