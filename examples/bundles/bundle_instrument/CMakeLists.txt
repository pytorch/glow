set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOW_BINARY_DIR}/bundles)
set(LENET_MNIST_BUNDLE_DIR ${GLOW_BINARY_DIR}/examples/bundles/bundle_instrument)
set(LENET_MNIST_GLOW_S3 "http://fb-glow-assets.s3.amazonaws.com/models/lenet_mnist")
set(LENET_MNIST_NET_FILES predict_net.pbtxt predict_net.pb init_net.pb)
set(MODEL_INPUT_NAME "data")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/mnist)

# Output directory.
set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/bundle_instrument)
set(INSTRUMENT_DEBUG_DIR ${BUNDLE_OUTPUT_DIRECTORY}/instrument-debug-data)
set(INSTRUMENT_IR_DIR ${BUNDLE_OUTPUT_DIRECTORY}/instrument-ir-data)
add_custom_target(BundleInstrumentDir ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE_OUTPUT_DIRECTORY}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${INSTRUMENT_DEBUG_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${INSTRUMENT_IR_DIR}
)

# Executable.
add_executable(BundleInstrument $<TARGET_OBJECTS:BundleInstrumentMain>)
set_target_properties(BundleInstrument PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BUNDLE_OUTPUT_DIRECTORY})
target_link_libraries(BundleInstrument ${BUNDLE_OUTPUT_DIRECTORY}/lenet_mnist.o png)
add_dependencies(BundleInstrument BundleInstrumentMain BundleInstrumentNet)

# Bundle.
add_custom_command(
  OUTPUT
    ${BUNDLE_OUTPUT_DIRECTORY}/lenet_mnist.o
  COMMAND
    model-compiler -g -model=${LENET_MNIST_BUNDLE_DIR}/lenet_mnist
    -model-input=${MODEL_INPUT_NAME},float,[1,1,28,28]
    -backend=CPU -emit-bundle=${BUNDLE_OUTPUT_DIRECTORY}
    -bundle-api=static
    -instrument-debug -instrument-debug-format=rawbin
    -instrument-debug-dir=${INSTRUMENT_DEBUG_DIR}
    -instrument-ir
  DEPENDS
    model-compiler BundleInstrumentDir
)
add_custom_target(BundleInstrumentNet DEPENDS ${BUNDLE_OUTPUT_DIRECTORY}/lenet_mnist.o BundleInstrumentNetFiles)

# Driver program with main function for regular bundle
add_library(BundleInstrumentMain OBJECT main.cpp)
target_compile_options(BundleInstrumentMain PRIVATE -std=c++11 -g)
target_include_directories(BundleInstrumentMain PUBLIC ${BUNDLE_OUTPUT_DIRECTORY})
target_compile_definitions(BundleInstrumentMain PUBLIC INSTRUMENT_IR_DIR="${INSTRUMENT_IR_DIR}")
add_dependencies(BundleInstrumentMain BundleInstrumentNet)

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
add_custom_target(BundleInstrumentNetFiles DEPENDS ${LENET_MNIST_NET_FILES})
