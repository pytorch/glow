set(LENET_MNIST_BUNDLE_DIR ${GLOW_BINARY_DIR}/examples/bundles/lenet_mnist)
set(LENET_MNIST_GLOW_S3 "http://fb-glow-assets.s3.amazonaws.com/models/lenet_mnist")
set(LENET_MNIST_NET_FILES predict_net.pbtxt predict_net.pb init_net.pb)
set(MODEL_INPUT_NAME "data")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/mnist)
set(RUNTIME_OUTPUT_DIRECTORY ${LENET_MNIST_BUNDLE_DIR})

# Run Commands
# ============
# Regular
add_custom_command(
  OUTPUT
    RunBundleCommand
  COMMAND
    LeNetMnistBundle ${IMAGES}/*.png
  DEPENDS
    LeNetMnistBundle
)
add_custom_target(RunLeNetMnistBundle DEPENDS RunBundleCommand LeNetMnistBundle)

# Quantized
add_custom_command(
  OUTPUT
    RunQuantizedBundleCommand
  COMMAND
    QuantizedLeNetMnistBundle ${IMAGES}/*.png
  DEPENDS
    QuantizedLeNetMnistBundle
)
add_custom_target(RunQuantizedLeNetMnistBundle DEPENDS RunQuantizedBundleCommand QuantizedLeNetMnistBundle)

# Final Executables
# =================
# Regular
add_executable(LeNetMnistBundle $<TARGET_OBJECTS:LeNetMnistBundleMain>)
target_link_libraries(LeNetMnistBundle ${LENET_MNIST_BUNDLE_DIR}/lenet_mnist.o png)
add_dependencies(LeNetMnistBundle LeNetMnistBundleMain LeNetMnistBundleNet)

# Quantized
add_executable(QuantizedLeNetMnistBundle $<TARGET_OBJECTS:LeNetMnistBundleMain>)
target_link_libraries(QuantizedLeNetMnistBundle ${LENET_MNIST_BUNDLE_DIR}/quantized_lenet_mnist.o png)
add_dependencies(QuantizedLeNetMnistBundle LeNetMnistBundleMain QuantizedLeNetMnistBundleNet)

# Glow Bundles
# ============
# Regular Bundle
add_custom_command(
  OUTPUT
    ${LENET_MNIST_BUNDLE_DIR}/lenet_mnist.o
  COMMAND
    image-classifier ${IMAGES}/3_1020.png -g -image-mode=0to1
    -m=${LENET_MNIST_BUNDLE_DIR}/lenet_mnist -model-input-name=${MODEL_INPUT_NAME}
    -cpu -emit-bundle ${LENET_MNIST_BUNDLE_DIR}
  DEPENDS
    image-classifier
)
add_custom_target(LeNetMnistBundleNet DEPENDS ${LENET_MNIST_BUNDLE_DIR}/lenet_mnist.o LeNetMnistBundleNetFiles)

# Quantization Profile
add_custom_command(
  OUTPUT
    ${LENET_MNIST_BUNDLE_DIR}/profile.yml
  COMMAND
    image-classifier ${IMAGES}/*.png -image-mode=0to1
    -dump-profile=${LENET_MNIST_BUNDLE_DIR}/profile.yml
    -m=${LENET_MNIST_BUNDLE_DIR}/lenet_mnist -model-input-name=${MODEL_INPUT_NAME}
  DEPENDS
    image-classifier
)
add_custom_target(LeNetMnistBundleQuantizationProfile DEPENDS ${LENET_MNIST_BUNDLE_DIR}/profile.yml LeNetMnistBundleNetFiles)

# Quantized Bundle
add_custom_command(
  OUTPUT
    ${LENET_MNIST_BUNDLE_DIR}/quantized_lenet_mnist.o
  COMMAND
    image-classifier ${IMAGES}/3_1020.png -g -image-mode=0to1 -load-profile=profile.yml
    -m=${LENET_MNIST_BUNDLE_DIR}/lenet_mnist -model-input-name=${MODEL_INPUT_NAME}
    -cpu -emit-bundle ${LENET_MNIST_BUNDLE_DIR} && mv ${LENET_MNIST_BUNDLE_DIR}/lenet_mnist.o ${LENET_MNIST_BUNDLE_DIR}/quantized_lenet_mnist.o
  DEPENDS
    image-classifier
)
add_custom_target(QuantizedLeNetMnistBundleNet DEPENDS ${LENET_MNIST_BUNDLE_DIR}/quantized_lenet_mnist.o LeNetMnistBundleQuantizationProfile)

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

unset(RUNTIME_OUTPUT_DIRECTORY)
