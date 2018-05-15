#!/usr/bin/env bash

./bin/image-classifier tests/images/imagenet/*.png -image_mode=0to1 -m=resnet50 $@
./bin/image-classifier tests/images/imagenet/*.png -image_mode=128to127 -m=vgg19 $@
./bin/image-classifier tests/images/imagenet/*.png -image_mode=128to127 -m=squeezenet $@
./bin/image-classifier tests/images/imagenet/*.png -image_mode=0to256 -m=zfnet512 $@
./bin/image-classifier tests/images/imagenet/*.png -image_mode=0to1 -m=densenet121 $@
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier $png_filename -image_mode=0to1 -m=resnet50/model.onnx $@
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier $png_filename -image_mode=128to127 -m=vgg19/model.onnx $@
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier $png_filename -image_mode=128to127 -m=squeezenet/model.onnx $@
done

./bin/image-classifier tests/images/mnist/*.png -image_mode=0to1 -m=lenet_mnist $@
