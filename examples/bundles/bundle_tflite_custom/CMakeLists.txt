set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOW_BINARY_DIR}/bundles)

# Output directory.
set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/bundle_tflite_custom)
add_custom_target(BundleTFLiteCustomDir ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE_OUTPUT_DIRECTORY}
)

# Executable.
add_executable(BundleTFLiteCustom $<TARGET_OBJECTS:BundleTFLiteCustomMain>)
set_target_properties(BundleTFLiteCustom PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BUNDLE_OUTPUT_DIRECTORY})
target_link_libraries(BundleTFLiteCustom ${BUNDLE_OUTPUT_DIRECTORY}/model.o)
add_dependencies(BundleTFLiteCustom BundleTFLiteCustomMain BundleTFLiteCustomNet)

# Bundle.
add_custom_command(
  OUTPUT
    ${BUNDLE_OUTPUT_DIRECTORY}/model.o
  COMMAND
    model-compiler
    -g
    -model=${CMAKE_CURRENT_SOURCE_DIR}/tflite_custom.tflite
    -network-name=model
    -backend=CPU
    -emit-bundle=${BUNDLE_OUTPUT_DIRECTORY}
    -bundle-api=static
  DEPENDS
    model-compiler BundleTFLiteCustomDir
)
add_custom_target(BundleTFLiteCustomNet DEPENDS ${BUNDLE_OUTPUT_DIRECTORY}/model.o)

# Driver program with main function for regular bundle
add_library(BundleTFLiteCustomMain OBJECT main.cpp)
target_compile_options(BundleTFLiteCustomMain PRIVATE -std=c++11 -g)
target_include_directories(BundleTFLiteCustomMain PUBLIC ${BUNDLE_OUTPUT_DIRECTORY} ${GLOW_THIRDPARTY_DIR}/tflite)
add_dependencies(BundleTFLiteCustomMain BundleTFLiteCustomNet)
