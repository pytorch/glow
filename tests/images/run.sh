#!/usr/bin/env bash

./bin/image-classifier tests/images/imagenet/*.png -image_mode=0to1 -m=resnet50 "$@"
./bin/image-classifier tests/images/imagenet/*.png -image_mode=neg128to127 -m=vgg19 "$@"
./bin/image-classifier tests/images/imagenet/*.png -image_mode=neg128to127 -m=squeezenet "$@"
./bin/image-classifier tests/images/imagenet/*.png -image_mode=0to256 -m=zfnet512 "$@"
./bin/image-classifier tests/images/imagenet/*.png -image_mode=0to1 -m=densenet121 "$@"
./bin/image-classifier tests/images/imagenet/*.png -image_mode=0to1 -m=shufflenet "$@"
./bin/image-classifier tests/images/mnist/*.png -image_mode=0to1 -m=lenet_mnist "$@"
./bin/image-classifier tests/images/imagenet/*.png -image_mode=0to256 -m=inception_v1 "$@"
./bin/image-classifier tests/images/imagenet/*.png -image_mode=0to256 -m=bvlc_alexnet "$@"
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image_mode=0to1 -m=resnet50/model.onnx "$@"
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image_mode=neg128to127 -m=vgg19/model.onnx "$@"
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image_mode=neg128to127 -m=squeezenet/model.onnx "$@"
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image_mode=0to256 -m=zfnet512/model.onnx "$@"
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image_mode=0to1 -m=densenet121/model.onnx "$@"
done
./bin/image-classifier tests/images/imagenet/zebra_340.png -image_mode=0to1 -m=shufflenet/model.onnx "$@"
for png_filename in tests/images/mnist/*.png; do
  ./bin/image-classifier "$png_filename" -image_mode=0to1 -m=mnist.onnx "$@"
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image_mode=0to256 -m=inception_v1/model.onnx "$@"
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image_mode=0to256 -m=bvlc_alexnet/model.onnx "$@"
done
