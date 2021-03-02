set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOW_BINARY_DIR}/bundles)
set(LENET_MNIST_BUNDLE_DIR ${GLOW_BINARY_DIR}/examples/bundles/bundle_with_extra_objects)
set(LENET_MNIST_GLOW_S3 "http://fb-glow-assets.s3.amazonaws.com/models/lenet_mnist")
set(LENET_MNIST_NET_FILES predict_net.pbtxt predict_net.pb init_net.pb)
set(MODEL_INPUT_NAME "data")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/mnist)

# Output directory.
set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/bundle_with_extra_objects)
add_custom_target(BundleWithExtraObjectsDir ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE_OUTPUT_DIRECTORY}
)

# Executable.
add_executable(BundleWithExtraObjects $<TARGET_OBJECTS:BundleWithExtraObjectsMain>)
set_target_properties(BundleWithExtraObjects PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BUNDLE_OUTPUT_DIRECTORY})
target_link_libraries(BundleWithExtraObjects ${BUNDLE_OUTPUT_DIRECTORY}/lenet_mnist.o)
add_dependencies(BundleWithExtraObjects BundleWithExtraObjectsMain BundleWithExtraObjectsNet)

# Bundle.
add_custom_command(
  OUTPUT
    ${BUNDLE_OUTPUT_DIRECTORY}/lenet_mnist.o
  COMMAND
    model-compiler -model=${LENET_MNIST_BUNDLE_DIR}/lenet_mnist
    -model-input=${MODEL_INPUT_NAME},float,[1,1,28,28]
    -backend=CPU -emit-bundle=${BUNDLE_OUTPUT_DIRECTORY}
    -bundle-api=static
    -bundle-objects=test.o
  DEPENDS
    model-compiler BundleWithExtraObjectsDir
)
add_custom_target(BundleWithExtraObjectsNet DEPENDS ${BUNDLE_OUTPUT_DIRECTORY}/lenet_mnist.o BundleWithExtraObjectsNetFiles)

# Driver program with main function for regular bundle
add_library(BundleWithExtraObjectsMain OBJECT main.cpp)
target_compile_options(BundleWithExtraObjectsMain PRIVATE -std=c++11 -g)
target_include_directories(BundleWithExtraObjectsMain PUBLIC ${BUNDLE_OUTPUT_DIRECTORY})
add_dependencies(BundleWithExtraObjectsMain BundleWithExtraObjectsNet)

# Network structure and weight files
foreach(file ${LENET_MNIST_NET_FILES})
    add_custom_command(
    OUTPUT
      ${file}
    COMMAND
      wget
    ARGS
      "${LENET_MNIST_GLOW_S3}/${file}" -P ${LENET_MNIST_BUNDLE_DIR}/lenet_mnist -nc
)
endforeach()
add_custom_target(BundleWithExtraObjectsNetFiles DEPENDS ${LENET_MNIST_NET_FILES})
