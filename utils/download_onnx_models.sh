#!/usr/bin/env bash

for modelname in resnet50 vgg19 squeezenet zfnet512 densenet121 shufflenet; do
  wget -nc https://s3.amazonaws.com/download.onnx/models/opset_6/$modelname.tar.gz
  tar -xzvf $modelname.tar.gz
done

for modelname in inception_v1 inception_v2 bvlc_alexnet; do
  wget -nc https://s3.amazonaws.com/download.onnx/models/opset_8/$modelname.tar.gz
  tar -xzvf $modelname.tar.gz
done

# shellcheck disable=SC2043
for modelname in lenet_mnist; do
  wget -nc http://fb-glow-assets.s3.amazonaws.com/models/$modelname.tar.gz
  tar -xzvf $modelname.tar.gz
done
