#!/usr/bin/env bash

./bin/image-classifier tests/images/imagenet/*.png -image_mode=0to1 -m=resnet50 -model_input_name=gpu_0/data "$@"
./bin/image-classifier tests/images/imagenet/*.png -image_mode=neg128to127 -m=vgg19 -model_input_name=data "$@"
./bin/image-classifier tests/images/imagenet/*.png -image_mode=neg128to127 -m=squeezenet -model_input_name=data "$@"
./bin/image-classifier tests/images/imagenet/*.png -image_mode=0to255 -m=zfnet512 -model_input_name=gpu_0/data "$@"
./bin/image-classifier tests/images/imagenet/*.png -image_mode=0to1 -m=densenet121 -model_input_name=data -compute_softmax "$@"
./bin/image-classifier tests/images/imagenet/*.png -image_mode=0to1 -m=shufflenet -model_input_name=gpu_0/data "$@"
./bin/image-classifier tests/images/mnist/*.png -image_mode=0to1 -m=lenet_mnist -model_input_name=data "$@"
./bin/image-classifier tests/images/imagenet/*.png -image_mode=0to255 -m=inception_v1 -model_input_name=data "$@"
./bin/image-classifier tests/images/imagenet/*.png -image_mode=0to255 -m=bvlc_alexnet -model_input_name=data "$@"
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image_mode=0to1 -m=resnet50/model.onnx -model_input_name=gpu_0/data_0 "$@"
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image_mode=neg128to127 -m=vgg19/model.onnx -model_input_name=data_0 "$@"
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image_mode=neg128to127 -m=squeezenet/model.onnx -model_input_name=data_0 "$@"
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image_mode=0to255 -m=zfnet512/model.onnx -model_input_name=gpu_0/data_0 "$@"
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image_mode=0to1 -m=densenet121/model.onnx -model_input_name=data_0 -compute_softmax "$@"
done
./bin/image-classifier tests/images/imagenet/zebra_340.png -image_mode=0to1 -m=shufflenet/model.onnx -model_input_name=gpu_0/data_0 "$@"
for png_filename in tests/images/mnist/*.png; do
  ./bin/image-classifier "$png_filename" -image_mode=0to1 -m=mnist.onnx -model_input_name=data_0 "$@"
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image_mode=0to255 -m=inception_v1/model.onnx -model_input_name=data_0 "$@"
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image_mode=0to255 -m=bvlc_alexnet/model.onnx -model_input_name=data_0 "$@"
done
#Inference test for TF ONNX models
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image_mode=0to1 -m=googlenet_v1_slim/googlenet_v1_slim.onnx -model_input_name=input:0 -image_layout=NHWC -label_offset=1 "$@"
done
for png_filename in tests/images/imagenet_299/*.png; do
  ./bin/image-classifier "$png_filename" -image_mode=0to1 -m=googlenet_v4_slim/googlenet_v4_slim.onnx -model_input_name=input:0 -image_layout=NHWC -label_offset=1 "$@"
done
