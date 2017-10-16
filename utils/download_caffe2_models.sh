#!/usr/bin/env bash

for model in "resnet50" "vgg16" "vgg19" "inception_v1" "inception_v2" "shufflenet" "squeezenet" ; do
  for file in predict_net.pbtxt predict_net.pb init_net.pb; do
    wget https://s3.amazonaws.com/download.caffe2.ai/models/$model/$file -P $model
  done
done


