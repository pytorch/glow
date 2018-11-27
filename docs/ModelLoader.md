## Model Loader

This document describes how to load a pretrained model from other deep learning framework, such as PyTorch and Caffe2, to Glow and then execute inside.

### Examples

#### PyTorch

Since Glow supports loading ONNX format model directly, we can export models of PyTorch into the ONNX format, and then load the ONNX model to Glow.

There are many pretrained models provided by [torchvision package](https://pytorch.org/docs/stable/torchvision/models.html). Following is the model list:

ImageNet 1-crop error rates (224x224)

| Network | Top-1 error| Top-5 error|
| :------ | :------ | :------ |
|alexnet          |                 43.45        |   20.91|
|vgg11             |               30.98         |  11.37|
|vgg13               |             30.07          | 10.75|
|vgg16                |            28.41          | 9.62|
|vgg19                 |           27.62          | 9.12|
|vgg11_bn |  29.62 |          10.19|
|vgg13_bn  | 28.45   |        9.63|
|vgg16_bn  | 26.63     |      8.50|
|vgg19_bn  | 25.76       |    8.15|
|resnet18                    |     30.24         |  10.92|
|resnet34                     |    26.70          | 8.58|
|resnet50                     |   23.85         |  7.13|
|resnet101              |   22.63         |  6.44|
|resnet152              |  21.69         |  5.94|
|squeezenet1_0               |    41.90        |   19.58|
|squeezenet1_1                |   41.81         |  19.38|
|dense121                   | 25.35          | 7.83|
|desnet169                     |24.00          | 7.00|
|desnet201                     | 22.80          | 6.43|
|desnet161                      |22.35          | 6.20|
|inception_v3                     | 22.55          | 6.44|

We provide an util function to export ONNX model from PyTorch pretrained model. Please refer to [export_onnx_model.py](../utils/export_onnx_model.py)

This util function actually calls functions in [torch.onnx module](https://pytorch.org/docs/0.3.1/onnx.html).

Following are some examples of using the util function:

```bash
# You can set your own glow path.
export GLOW_PATH=~/glow/

cd ${GLOW_PATH}/utils
# This command export resnet18 model from PyTorch, and save the model as resnet18.onnx in current directory
python export_onnx_model.py --model_name resnet18 --model_path resnet18.onnx --model_input_name data

# Export alexnet model
python export_onnx_model.py --model_name alexnet --model_path alexnet.onnx --model_input_name data
```


After we get the ONNX model, we can load it and run it in Glow.

```bash
# Please keep the model-input-name same with export_onnx_model util function
${GLOW_PATH}/build/bin/image-classifier ${GLOW_PATH}/tests/images/imagenet/cat_285.png -image-mode=0to1 -m resnet18.onnx -model-input-name=data -cpu
```

The running results can be:

```bash
Model: resnet18.onnx
 File: ${GLOW_PATH}/tests/images/imagenet/cat_285.png   Label-K1: 285 (probability: 16.4305)
```


