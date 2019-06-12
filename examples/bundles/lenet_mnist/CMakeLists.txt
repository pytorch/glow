set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOW_BINARY_DIR}/bundles)
set(LENET_MNIST_BUNDLE_DIR ${GLOW_BINARY_DIR}/examples/bundles/lenet_mnist)
set(LENET_MNIST_GLOW_S3 "http://fb-glow-assets.s3.amazonaws.com/models/lenet_mnist")
set(LENET_MNIST_NET_FILES predict_net.pbtxt predict_net.pb init_net.pb)
set(MODEL_INPUT_NAME "data")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/mnist)

# Output directories for regular and quantized outputs
set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lenet_mnist)
set(QUANTIZED_BUNDLE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/quantized_lenet_mnist)

add_custom_target(LeNetMnistBundleDir ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE_OUTPUT_DIRECTORY}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY}
)

# Final Executables
# =================
# Regular
add_executable(LeNetMnistBundle $<TARGET_OBJECTS:LeNetMnistBundleMain>)
set_target_properties(LeNetMnistBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BUNDLE_OUTPUT_DIRECTORY})
target_link_libraries(LeNetMnistBundle ${BUNDLE_OUTPUT_DIRECTORY}/lenet_mnist.o png)
add_dependencies(LeNetMnistBundle LeNetMnistBundleMain LeNetMnistBundleNet)

# Quantized
add_executable(QuantizedLeNetMnistBundle $<TARGET_OBJECTS:LeNetMnistBundleMain>)
set_target_properties(QuantizedLeNetMnistBundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY})
target_link_libraries(QuantizedLeNetMnistBundle ${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY}/lenet_mnist.o png)
add_dependencies(QuantizedLeNetMnistBundle LeNetMnistBundleMain QuantizedLeNetMnistBundleNet)

# Glow Bundles
# ============
# Regular Bundle
add_custom_command(
  OUTPUT
    ${BUNDLE_OUTPUT_DIRECTORY}/lenet_mnist.o
  COMMAND
    image-classifier ${IMAGES}/3_1020.png -g -i=0to1
    -m=${LENET_MNIST_BUNDLE_DIR}/lenet_mnist -model-input-name=${MODEL_INPUT_NAME}
    -backend=CPU -emit-bundle ${BUNDLE_OUTPUT_DIRECTORY}
  DEPENDS
    image-classifier LeNetMnistBundleDir
)
add_custom_target(LeNetMnistBundleNet DEPENDS ${BUNDLE_OUTPUT_DIRECTORY}/lenet_mnist.o LeNetMnistBundleNetFiles)

# Quantization Profile
add_custom_command(
  OUTPUT
    ${LENET_MNIST_BUNDLE_DIR}/profile.yml
  COMMAND
    image-classifier ${IMAGES}/*.png -i=0to1
    -dump-profile=${LENET_MNIST_BUNDLE_DIR}/profile.yml
    -m=${LENET_MNIST_BUNDLE_DIR}/lenet_mnist -model-input-name=${MODEL_INPUT_NAME}
  DEPENDS
    image-classifier
)
add_custom_target(LeNetMnistBundleQuantizationProfile DEPENDS ${LENET_MNIST_BUNDLE_DIR}/profile.yml LeNetMnistBundleNetFiles)

# Quantized Bundle
add_custom_command(
  OUTPUT
  ${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY}/lenet_mnist.o
  COMMAND
    image-classifier ${IMAGES}/3_1020.png -g -i=0to1 -load-profile=profile.yml
    -m=${LENET_MNIST_BUNDLE_DIR}/lenet_mnist -model-input-name=${MODEL_INPUT_NAME}
    -backend=CPU -emit-bundle ${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY}
  DEPENDS
  image-classifier LeNetMnistBundleDir
)
add_custom_target(QuantizedLeNetMnistBundleNet DEPENDS ${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY}/lenet_mnist.o LeNetMnistBundleQuantizationProfile)

# Other
# =====
# Driver program with main function
add_library(LeNetMnistBundleMain OBJECT main.cpp)
target_compile_options(LeNetMnistBundleMain PRIVATE -std=c++11 -g)

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
add_custom_target(LeNetMnistBundleNetFiles DEPENDS ${LENET_MNIST_NET_FILES})
