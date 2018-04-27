#!/usr/bin/env bash

./bin/loader tests/images/imagenet/*.png -image_mode=0to1 -d=resnet50 $@
./bin/loader tests/images/imagenet/*.png -image_mode=128to127 -d=vgg19 $@
./bin/loader tests/images/imagenet/*.png -image_mode=128to127 -d=squeezenet $@
./bin/loader tests/images/imagenet/*.png -image_mode=0to256 -d=vgg16 $@
for png_filename in tests/images/imagenet/*.png; do
  ./bin/loader $png_filename -image_mode=0to1 -o=resnet50/model.onnx $@
done

./bin/loader tests/images/mnist/*.png -image_mode=0to1 -d=lenet_mnist $@
