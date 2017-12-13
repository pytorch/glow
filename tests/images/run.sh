#!/usr/bin/env bash

for png_filename in tests/images/*.png; do
  ./bin/loader $png_filename -image_mode=0to1 -n=resnet50/predict_net.pb -w=resnet50/init_net.pb
  ./bin/loader $png_filename -image_mode=128to127 -n=vgg19/predict_net.pb -w=vgg19/init_net.pb
  ./bin/loader $png_filename -image_mode=128to127 -n=squeezenet/predict_net.pb -w=squeezenet/init_net.pb
  ./bin/loader $png_filename -image_mode=0to256 -n=vgg16/predict_net.pb -w=vgg16/init_net.pb
done

