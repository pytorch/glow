set(RESNET50_BUNDLE_DIR ${GLOW_BINARY_DIR}/examples/bundles/resnet50)
set(RESNET50_GLOW_S3 "http://fb-glow-assets.s3.amazonaws.com/models/resnet50")
set(RESNET50_NET_FILES predict_net.pbtxt predict_net.pb init_net.pb)
set(MODEL_INPUT_NAME "gpu_0/data")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/imagenet)
set(RUNTIME_OUTPUT_DIRECTORY ${RESNET50_BUNDLE_DIR})

# Run Commands
# ============
# Regular
add_custom_command(
  OUTPUT
    RunBundleCommand
  COMMAND
    ResNet50Bundle ${IMAGES}/*.png
  DEPENDS
    ResNet50Bundle
)
add_custom_target(RunResNet50Bundle DEPENDS RunBundleCommand ResNet50Bundle)

# Quantized
add_custom_command(
  OUTPUT
    RunQuantizedBundleCommand
  COMMAND
    QuantizedResNet50Bundle ${IMAGES}/*.png
  DEPENDS
    QuantizedResNet50Bundle
)
add_custom_target(RunQuantizedResNet50Bundle DEPENDS RunQuantizedBundleCommand QuantizedResNet50Bundle)

# Final Executables
# =================
# Regular
add_executable(ResNet50Bundle $<TARGET_OBJECTS:ResNet50BundleMain>)
set_target_properties(ResNet50Bundle PROPERTIES LINK_FLAGS "-lpng")
target_link_libraries(ResNet50Bundle ${RESNET50_BUNDLE_DIR}/resnet50.o)
add_dependencies(ResNet50Bundle ResNet50BundleMain ResNet50BundleNet)

# Quantized
add_executable(QuantizedResNet50Bundle $<TARGET_OBJECTS:ResNet50BundleMain>)
set_target_properties(QuantizedResNet50Bundle PROPERTIES LINK_FLAGS "-lpng")
target_link_libraries(QuantizedResNet50Bundle ${RESNET50_BUNDLE_DIR}/quantized_resnet50.o)
add_dependencies(QuantizedResNet50Bundle ResNet50BundleMain QuantizedResNet50BundleNet)

# Glow Bundles
# ============
# Regular Bundle
add_custom_command(
  OUTPUT
    ${RESNET50_BUNDLE_DIR}/resnet50.o
  COMMAND
    image-classifier ${IMAGES}/dog_207.png -g -image_mode=0to1
    -m=${RESNET50_BUNDLE_DIR}/resnet50 -model_input_name=${MODEL_INPUT_NAME}
    -cpu -emit-bundle ${RESNET50_BUNDLE_DIR} 
  DEPENDS 
    image-classifier
)
add_custom_target(ResNet50BundleNet DEPENDS ${RESNET50_BUNDLE_DIR}/resnet50.o ResNet50BundleNetFiles)

# Quantization Profile
add_custom_command(
  OUTPUT
    ${RESNET50_BUNDLE_DIR}/profile.yml
  COMMAND
    image-classifier ${IMAGES}/*.png -image_mode=0to1
    -dump_profile=${RESNET50_BUNDLE_DIR}/profile.yml
    -m=${RESNET50_BUNDLE_DIR}/resnet50 -model_input_name=${MODEL_INPUT_NAME}
  DEPENDS
    image-classifier
)
add_custom_target(ResNet50BundleQuantizationProfile DEPENDS ${RESNET50_BUNDLE_DIR}/profile.yml ResNet50BundleNetFiles)

# Quantized Bundle
add_custom_command(
  OUTPUT
    ${RESNET50_BUNDLE_DIR}/quantized_resnet50.o
  COMMAND
    image-classifier ${IMAGES}/dog_207.png -g -image_mode=0to1 -load_profile=profile.yml
    -m=${RESNET50_BUNDLE_DIR}/resnet50 -model_input_name=${MODEL_INPUT_NAME}
    -cpu -emit-bundle ${RESNET50_BUNDLE_DIR} && mv ${RESNET50_BUNDLE_DIR}/resnet50.o ${RESNET50_BUNDLE_DIR}/quantized_resnet50.o
  DEPENDS
    image-classifier
)
add_custom_target(QuantizedResNet50BundleNet DEPENDS ${RESNET50_BUNDLE_DIR}/quantized_resnet50.o ResNet50BundleQuantizationProfile)

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

unset(RUNTIME_OUTPUT_DIRECTORY)
