set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOW_BINARY_DIR}/bundles)
set(RESNET50_BUNDLE_DIR ${GLOW_BINARY_DIR}/examples/bundles/resnet50)
set(RESNET50_GLOW_S3 "http://fb-glow-assets.s3.amazonaws.com/models/resnet50")
set(RESNET50_NET_FILES predict_net.pbtxt predict_net.pb init_net.pb)
set(MODEL_INPUT_NAME "gpu_0/data")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/imagenet)

# Output directories for regular and quantized outputs
set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resnet50)
set(QUANTIZED_BUNDLE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/quantized_resnet50)

add_custom_target(ResNet50BundleDir ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE_OUTPUT_DIRECTORY}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY}
)

# Final Executables
# =================
# Regular
add_executable(ResNet50Bundle $<TARGET_OBJECTS:ResNet50BundleMain>)
set_target_properties(ResNet50Bundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BUNDLE_OUTPUT_DIRECTORY})
target_link_libraries(ResNet50Bundle ${BUNDLE_OUTPUT_DIRECTORY}/resnet50.o png)
add_dependencies(ResNet50Bundle ResNet50BundleMain ResNet50BundleNet)

# Quantized
add_executable(QuantizedResNet50Bundle $<TARGET_OBJECTS:ResNet50BundleMain>)
set_target_properties(QuantizedResNet50Bundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY})
target_link_libraries(QuantizedResNet50Bundle ${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY}/resnet50.o png)
add_dependencies(QuantizedResNet50Bundle ResNet50BundleMain QuantizedResNet50BundleNet)

# Glow Bundles
# ============
# Regular Bundle
add_custom_command(
  OUTPUT
    ${BUNDLE_OUTPUT_DIRECTORY}/resnet50.o
  COMMAND
    image-classifier ${IMAGES}/dog_207.png -g -image-mode=0to1
    -m=${RESNET50_BUNDLE_DIR}/resnet50 -model-input-name=${MODEL_INPUT_NAME}
    -backend=CPU -emit-bundle ${BUNDLE_OUTPUT_DIRECTORY}
  DEPENDS
    image-classifier ResNet50BundleDir
)
add_custom_target(ResNet50BundleNet DEPENDS ${BUNDLE_OUTPUT_DIRECTORY}/resnet50.o ResNet50BundleNetFiles)

# Quantization Profile
add_custom_command(
  OUTPUT
    ${RESNET50_BUNDLE_DIR}/profile.yml
  COMMAND
    image-classifier ${IMAGES}/*.png -i=0to1
    -dump-profile=${RESNET50_BUNDLE_DIR}/profile.yml
    -m=${RESNET50_BUNDLE_DIR}/resnet50 -model-input-name=${MODEL_INPUT_NAME}
  DEPENDS
  image-classifier
)
add_custom_target(ResNet50BundleQuantizationProfile DEPENDS ${RESNET50_BUNDLE_DIR}/profile.yml ResNet50BundleNetFiles)

# Quantized Bundle
add_custom_command(
  OUTPUT
    ${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY}/resnet50.o
  COMMAND
    image-classifier ${IMAGES}/dog_207.png -g -i=0to1 -load-profile=profile.yml -assert-all-nodes-quantized -keep-original-precision-for-nodes=SoftMax
    -m=${RESNET50_BUNDLE_DIR}/resnet50 -model-input-name=${MODEL_INPUT_NAME}
    -backend=CPU -emit-bundle ${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY}
  DEPENDS
  image-classifier ResNet50BundleDir
)
add_custom_target(QuantizedResNet50BundleNet DEPENDS ${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY}/resnet50.o ResNet50BundleQuantizationProfile)

# Other
# =====
# Driver program with main function
add_library(ResNet50BundleMain OBJECT main.cpp)
target_compile_options(ResNet50BundleMain PRIVATE -std=c++11 -g)

# Network structure and weight files
foreach(file ${RESNET50_NET_FILES})
    add_custom_command(
    OUTPUT
      ${file}
    COMMAND
      wget
    ARGS
      "${RESNET50_GLOW_S3}/${file}" -P ${RESNET50_BUNDLE_DIR}/resnet50 -nc
)
endforeach()
add_custom_target(ResNet50BundleNetFiles DEPENDS ${RESNET50_NET_FILES})
