#!/usr/bin/env bash

./bin/image-classifier tests/images/imagenet/*.png -image_mode=0to1 -m=resnet50 -model_input_name=gpu_0/data "$@"
./bin/image-classifier tests/images/imagenet/*.png -image_mode=neg128to127 -m=vgg19 -model_input_name=data "$@"
./bin/image-classifier tests/images/imagenet/*.png -image_mode=neg128to127 -m=squeezenet -model_input_name=data "$@"
./bin/image-classifier tests/images/imagenet/*.png -image_mode=0to256 -m=zfnet512 -model_input_name=gpu_0/data "$@"
./bin/image-classifier tests/images/imagenet/*.png -image_mode=0to1 -m=densenet121 -model_input_name=data "$@"
./bin/image-classifier tests/images/imagenet/*.png -image_mode=0to1 -m=shufflenet -model_input_name=gpu_0/data "$@"
./bin/image-classifier tests/images/mnist/*.png -image_mode=0to1 -m=lenet_mnist -model_input_name=data "$@"
./bin/image-classifier tests/images/imagenet/*.png -image_mode=0to256 -m=inception_v1 -model_input_name=data "$@"
./bin/image-classifier tests/images/imagenet/*.png -image_mode=0to256 -m=bvlc_alexnet -model_input_name=data "$@"
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
  ./bin/image-classifier "$png_filename" -image_mode=0to256 -m=zfnet512/model.onnx -model_input_name=gpu_0/data_0 "$@"
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image_mode=0to1 -m=densenet121/model.onnx -model_input_name=data_0 "$@"
done
./bin/image-classifier tests/images/imagenet/zebra_340.png -image_mode=0to1 -m=shufflenet/model.onnx -model_input_name=gpu_0/data_0 "$@"
for png_filename in tests/images/mnist/*.png; do
  ./bin/image-classifier "$png_filename" -image_mode=0to1 -m=mnist.onnx -model_input_name=data_0 "$@"
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image_mode=0to256 -m=inception_v1/model.onnx -model_input_name=data_0 "$@"
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image_mode=0to256 -m=bvlc_alexnet/model.onnx -model_input_name=data_0 "$@"
done
