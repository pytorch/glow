#!/usr/bin/env bash

for png_filename in tests/images/*.png; do
  ./bin/loader $png_filename 0to1 resnet50/predict_net.pb resnet50/init_net.pb
  ./bin/loader $png_filename 128to127 vgg19/predict_net.pb vgg19/init_net.pb
  ./bin/loader $png_filename 128to127 squeezenet/predict_net.pb squeezenet/init_net.pb
done

