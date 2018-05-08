#!/usr/bin/env bash

for modelname in resnet50 vgg19 squeezenet; do
  wget -nc https://s3.amazonaws.com/download.onnx/models/opset_6/$modelname.tar.gz
  tar -xzvf $modelname.tar.gz
done
