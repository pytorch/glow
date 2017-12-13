#!/usr/bin/env bash

for png_filename in tests/images/*.png; do
  ./bin/loader $png_filename -image_mode=0to1 -d=resnet50
  ./bin/loader $png_filename -image_mode=128to127 -d=vgg19
  ./bin/loader $png_filename -image_mode=128to127 -d=squeezenet
  ./bin/loader $png_filename -image_mode=0to256 -d=vgg16
done

