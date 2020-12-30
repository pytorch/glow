# Lenet Model for MNIST using Custom Ops

This example demonstrates how to run a model with custom ops on Interpreter
backend. We build a simple Lenet model with custom operations for the activation
and pooling layers. A scaled Tanh function is used as activation and pooling
layer uses L2 reduction.

These operations can easily be implemented via existing pytorch operations,
but we implement as new TorchScript operations for the sake of this example.

## Export ONNX model with custom ops

Run the `gen_lenet_custom.py` script to build the lenet model in pytorch using
TorchScript custom ops, and export an ONNX model with these custom ops present.

The script requires pytorch 1.6+, Ninja, C++ compiler with C++14 support.

> Note: for purpose of training and tracing the onnx model, the script
downloads a compressed MNIST dataset. It runs 5 epochs of training on this
dataset which takes few minutes.

```
python gen_custom_lenet.py
```

The script generates `lenet_mnist_custom.onnx` file in the current dir.

## Build the Custom Op shared objects for glow

`ScaledTanh.cpp` and `L2Pool.cpp` contain the implementation of custom op functions
and Interpreter execution kernels for glow.
Run following commands to compile the shared objects using G++.

```
g++ -shared -fPIC -std=c++11 -I../../include/glow/CustomOp -o libL2Pool.so L2Pool.cpp
g++ -shared -fPIC -std=c++11 -I../../include/glow/CustomOp -lm -o libScaledTanh.so ScaledTanh.cpp
```

## Execute using image-classifier

```
image-classifier -m lenet_mnist_custom.onnx -model-input input.1 -image-mode=0to1 -register-custom-op LenetCustomConfig.yaml tests/images/mnist/*.png
```
