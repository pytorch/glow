# Inference Engines
- [Inference Engines](#inference-engines)
  - [Introduction](#introduction)
  - [X-Inference-Engine](#x-inference-engine)
    - [Platform Support](#platform-support)
    - [Building and Build Options](#building-and-build-options)
    - [Command Line Options](#command-line-options)
      - [Notes on Input and Output](#notes-on-input-and-output)
    - [Examples](#examples)
      - [Sample Input](#sample-input)
      - [Running with Statically-Linked Bundles](#running-with-statically-linked-bundles)
      - [Running with Dynamically-Loadable Bundles](#running-with-dynamically-loadable-bundles)
      - [Performance Monitoring (Linux Only)](#performance-monitoring-linux-only)
    - [Technical Notes and Notes on Dependencies](#technical-notes-and-notes-on-dependencies)
    - [A Note on the Initial Contribution](#a-note-on-the-initial-contribution)
#





## Introduction
Inference engines are supplied with Glow but are not part of the Glow build system. The 
engines are included in the separate directory `[GLOW ROOT DIR]/inference_engines` and must
be built separately. CMake files are provided to facilitate building of the engines, and 
the engines can be cross-compiled for any supported platform. These engines can either be
used as stand-alone applications, as part of a backend library powering custom engines, or
as examples guiding the development of custom engines. Some engines, or some parts and features
of some engines, are only supported on specific platforms (e.g. UNIX or Linux). This is made 
explicit where applicable in the present documentation. While this is a limitation, one could 
extend the provided engines to presently unsupported platforms using the current engines as 
a base. As always, contributions are encouraged and greatly appreciated!

## X-Inference-Engine

The `x-inference-engine` was conceived and developed as a complement to the `XModelRunner` -- 
the latter being a generic model compiler/runner distributed with Glow along with `ImageClassifier` and the generic `ModelLoader`. That is, the `x-inference-engine` is a generic (for the most part)
engine that is able to execute inference using bundles produced by Glow (e.g. the `XModelRunner` or `ImageClassifier`, or other model builders). It provides flexibility in terms of bundle handling: bundles can either be supplied at 
compile time and statically compiled into the engine, or the engine can be compiled with dynamic bundle linking support, in which case bundles (that are built as dynamically linked objects) can be supplied to the engine at run time and loaded as dynamically linked libraries. This engine comes with a backend library as well as a command line application. That is, the engine can either be used as a backend to develop other engines, or it can be used as a stand-alone application.

### Platform Support

The base functionality of the `x-inference-engine` is currently supported on all UNIX-like platforms, including OSX and Linux, but is not supported on Windows (its support for Windows is limited to UNIX/Linux virtual machines or simulators such as Cygwin or the Linux Subsystem for Windows, as well as any virtual machines that can host a UNIX-like guest OS). It can also be built and run in any UNIX-like virtual machine (it has been tested extensively with Linux running natively, or as a guest provided by VirtualBox). Finally, some functionality is currently limited to Linux only (e.g. performance monitoring). During the development of the engine, the decision to include this functionality was not clear; the main reason it was included is to (1) provide some rudimentary performance monitoring support (albeit only in Linux for now), and (2) serve the currently supported Linux-only feature as an example to guide further development of this functionality and its extension to other platforms.

### Building and Build Options

Before you begin, please make sure to read the notes on dependencies in [Notes on Dependencies](#technical-notes-and-notes-on-dependencies).

There is a `CMakeLists.txt` supplied with the engine. To build the engine in the `build` directory under, for example, `[GLOW ROOT]/inference_eingines/build`, execute the command
```
cmake -G Ninja -DCMAKE_BUILD_TYPE=[Debug/Release] [ADDITIONAL OPTIONS] ../
```
followed by
```
ninja all
```

Of course, `Ninja` can be substituted with any other supported build engine, such as `Unix Makefiles`. The additional options, some required and some optional, are given in the table below. Note also that cross-compilation can be facilitated
by the `TARGET_OPTIONS` and `TARGET_LINKER_FLAGS` options (see below for details). Of course, in this case, when compiling
with statically linked bundles, one must make sure that the bundles were also cross-compiled for the target platform.

| Option | Expected Values | Description | Required | Notes |
| :------ | :------ | :------ | :------ | :------ |
|`LIB_ONLY` | ON/OFF | Whether to build the backend library only (without command line application) | No (default is OFF) | |
|`BUILD_FOR_DYNAMIC_LINKAGE` | ON/OFF | Whether to build for dynamic linkage (bundle not specified at compile time, but must be specified at runtime and linked in dynamically -- bundle must be dynamically linkable) | No (default is ON) | |
|`LINK_LIBS_STATICALLY` | ON/OFF | Whether to link everything (e.g. `glibc`, `libm`, etc.) statically | No (default is ON) | Static linkage generally inflates object footprint, but may be required when compiling for platforms that may exhibit library version mismatch or might not even carry the required libraries |
|`ENABLE_PERF_MONITORING` | ON/OFF | Whether to enable performance monitoring | No (default is OFF) | Supported on Linux only |
|`TARGET_OPTIONS` | A string of additional compile options | Provides the ability to specify additional compiler options (e.g. include directories, sysroot, etc -- may be helpful when cross-compiling) | No |  |
|`TARGET_LINKER_FLAGS` | A string of additional linker options | Provides the ability to specify additional linker options (e.g. libraries, sysroot, etc -- may be helpful when cross-compiling) | No |  |
|`LINKED_BUNDLE` | The bundle path | Specifies the bundle to be compiled in when not compiling for dynamic linkage | Yes only when `BUILD_FOR_DYNAMIC_LINKAGE` is OFF |  |
|`LINKED_MODEL_NAME` | The model name | Specifies the model name (the base name for symbols in the specified bundle) | Yes only when `BUILD_FOR_DYNAMIC_LINKAGE` is OFF |  |

The produced output is the static library `xinfer`, and, if `LIB_ONLY` is set to OFF, the stand-alone executable `x-infer`.

### Command Line Options

The stand-alone application accepts many command line options, most of which are required. These are described in the table
below, and the documentation can be written to `stdout` with the `--help` option. The options may differ depending on whether the application was compiled with a statically-linked bundle, or
with support for dynamic bundle linkage. 

| Option | Expected Values | Description | Required | Notes |
| :------ | :------ | :------ | :------ | :------ |
|`output` | File name | Output to the specified file instead of `stdout` | No | If omitted, output is written to `stdout` |
|`infile` | File name | File containing input(s) in binary format | Yes | |
|`intype` | `F32`, `F16`, `I16`, or `I8` | Input data type | Yes | |
|`outtype` | `F32`, `F16`, `I16`, or `I8` | Output data type | Yes | |
|`inlen` | Integer | Input tensor length (e.g. if the tensor is of shape `2x3x4`, its length is `2 * 3 * 4 = 24` | Yes | |
|`outlen` | Integer | Output tensor length (e.g. if the tensor is of shape `2x3x4`, its length is `2 * 3 * 4 = 24` | Yes | Only single output tensor is currently supported |
|`inname` | String (no spaces) | Input tensor name | Yes | Only single named input tensors are currently supported |
|`outname` | String (no spaces) | Output tensor name | Yes | Only single named output tensors are currently supported |
|`perf` | Optionless Flag | Whether to output performance logs | No (default is NO) | Only supported on Linux, and only available when compiled with the `ENABLE_PERF_MONITORING` option |
|`perflog` | String (no spaces) | Performance log output filename (`stdout` if omitted) | No | Only supported on Linux, and only available when compiled with the `ENABLE_PERF_MONITORING` option |
|`model` | String (no spaces) | Model name (maximum 128 characters) | Yes | Only available when compiled with the `BUILD_FOR_DYNAMIC_LINKAGE` option |

The positional arguments are described below.

| Argument | Expected Values | Description | Required | Notes |
| :------ | :------ | :------ | :------ | :------ |
|Weights file | String | File name of the weights file | Yes | Specified as the first positional argument if the engine was compiled with a statically linked bundle; otherwise, this is the second positional argument |
|Bundle file | String | File name of the dynamically linked bundle | Yes | Specified as the first positional argument if the engine was compiled with support for dynamic bundle linkage; otherwise, not applicable |

#### Notes on Input and Output

Input(s) are given in a single binary file. The layout is a bytestream. The application is able to break the file up into individual inputs knowing the single input length (specified with the `inlen` and `intype`) option. For instance, if the input consists of a single floating 32bit point number, then the input file containing two inputs would consist of 8 bytes, the first four containing the first input, and the last four containing the last input.

The output is composed in a similar way as a bytestream saved into a single binary file. The order of output bytes follows the order of input bytes in the input file.

It is a responsibility of the developer to take care of issues such as byte ordering (e.g. big- vs little-endianness) in both, the input/output pairs and the weights file.

### Examples 

For the examples, we shall use the following fully connected feed forward network that has been trained to identify the sine curve on the interval `[0, 2pi]`. Everything above the curve is assigned the class `0`, while everything below the curve is assigned the class `1`. 

```Python
import numpy as np
import torch
import random
 
class FeedForwardNN(torch.nn.Module):
    def __init__(self, nx, nh, ny, depth, hidden_activation = torch.nn.ReLU(),
                 output_activation = torch.nn.Sigmoid(), seed = None, dtype = torch.FloatTensor):
        super().__init__()
     
        if seed is not None:
            torch.manual_seed(seed)
         
        self.params = torch.nn.ParameterList()
 
        self.layers = [torch.nn.Linear(nx, nh).type(dtype)]
        self.layers.append(hidden_activation)
         
        for _ in range(depth):
            self.layers.append(torch.nn.Linear(nh, nh).type(dtype))
            self.layers.append(hidden_activation)
             
        self.layers.append(torch.nn.Linear(nh, ny).type(dtype))
        self.layers.append(output_activation)
         
        for layer in self.layers:
            self.params.extend(layer.parameters())
         
    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
 
        return y
     
def train(module, loss, X, Y, num_iters, learning_rate = 1e-1):
    num_samples = X.shape[0]
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
 
    for _ in range(num_iters):
        optimizer.zero_grad()
        cost = loss(module(X), Y).sum() / num_samples 
        cost.backward()
        optimizer.step()
 
# Number of training and test samples.
m_train = 10000
m_test  = 5000
 
# Generate our data - points in 2D separated by a sine curve.
#
# We set the seed to make sure the experiment is reproduceable.
np.random.seed(1)
 
x_train = np.random.rand(m_train, 2) * 2 - 1
y_train = np.array([[0 if p[1] < 0.5 * np.sin(2 * np.pi * p[0]) else 1 for p in x_train]])
 
# Intentionally misclassify 5% of the points.
m_train_missed = m_train // 20
indices = random.sample(range(0, m_train), m_train_missed)
for index in indices:
    y_train[0, index] = 0 if y_train[0, index] != 0 else 1
 
# Set the structure of our network...
nx    = 2    # Dimension of input vectors.
nh    = 8    # Number of neurons in each hidden layer.
ny    = 1    # Dimension of the output vector.
depth = 2    # Number of hidden layers.
seed  = 0    # The seed for random initialization of the weights.
 
# Logistic regression loss function for computing the cost.
def logit_loss(yhat, y):
    return -(y * torch.log(yhat) + (1 - y) * torch.log(1 - yhat))
 
ffnn = FeedForwardNN(nx, nh, ny, depth, seed=seed)
 
# Convert numpy data into torch tensors...
torch_train_input = torch.from_numpy(x_train).float()
torch_train_output = torch.from_numpy(y_train.T).float()
 
# Train the network...
train(ffnn, logit_loss, torch_train_input, torch_train_output, 2000)
 
# Set training mode to False, since at this point we're only interested
# in inference.
ffnn.train(False)
 
# Create dummy input to go along with our model.
inp = torch.autograd.Variable(torch_test_input[:1])
 
input_names = ["in"]
output_names = ["out"]
 
torch.onnx.export(ffnn, inp, "ffnn.onnx", input_names=input_names,
                  output_names=output_names, verbose=True, export_params=True)
```

The sample output from ONNX is below.

```
graph(%in : Float(1, 2)
%1 : Float(8, 2)
%2 : Float(8)
%3 : Float(8, 8)
%4 : Float(8)
%5 : Float(8, 8)
%6 : Float(8)
%7 : Float(1, 8)
%8 : Float(1)) {
%9 : Float(1, 8) = onnx::Gemm[alpha=1, beta=1, transB=1](%in, %1, %2), scope: FeedForwardNN/Linear
%10 : Float(1, 8) = onnx::Relu(%9), scope: FeedForwardNN/ReLU
%11 : Float(1, 8) = onnx::Gemm[alpha=1, beta=1, transB=1](%10, %3, %4), scope: FeedForwardNN/ReLU
%12 : Float(1, 8) = onnx::Relu(%11), scope: FeedForwardNN/ReLU
%13 : Float(1, 8) = onnx::Gemm[alpha=1, beta=1, transB=1](%12, %5, %6), scope: FeedForwardNN/ReLU
%14 : Float(1, 8) = onnx::Relu(%13), scope: FeedForwardNN/ReLU
%15 : Float(1, 1) = onnx::Gemm[alpha=1, beta=1, transB=1](%14, %7, %8), scope: FeedForwardNN/ReLU
%out : Float(1, 1) = onnx::Sigmoid(%15), scope: FeedForwardNN/Sigmoid
return (%out);
}
```

What is important to note here is that the input tensor name is `in` and the output tensor name is `out`. Also note that the input tensor dimension is `<1, 2>`, and the output tensor dimension is `<1, 1>`. The output tensor carries the probability of belonging to class `0`. This information will be used later.

Now let us produce two versions of bundles: one for static linking, and one suitable for dynamic linking. Make sure to change all the relative paths below to reflect your setup (we assume that you have built Glow and the included builders, in particular the `x-model-builder`)

**Statically linkable bundle:**
```
(base) wyessen [~/dts/nn/xgmr/build/debug/osx] $ ./bin/x-model-builder -input-tensor-dims=1,2 -output-tensor-names=out -model-input-name=in -model=../../../../pytorch_tuts/ffnn.onnx -emit-bundle=../output/ -backend=CPU -main-entry-name=ffnn -network-name=ffnn_static

(base) wyessen [~/dts/nn/xgmr/build/debug/osx] $ ls ../output/
ffnn_static.o       ffnn_static.weights

(base) wyessen [~/dts/nn/xgmr/build/debug/osx] $ 
```

**Dynamically linkable bundle:**
```
(base) wyessen [~/dts/nn/xgmr/build/debug/osx] $ ./bin/x-model-builder -input-tensor-dims=1,2 -output-tensor-names=out -model-input-name=in -model=../../../../pytorch_tuts/ffnn.onnx -emit-bundle=../output/ -backend=CPU -main-entry-name=ffnn -network-name=ffnn_dynamic -llvm-compiler=/usr/local/opt/llvm@8/bin/clang++ -llvm-compiler-opt=-shared

warning: overriding the module target triple with x86_64-apple-macosx10.14.0 [-Woverride-module]
1 warning generated.

(base) wyessen [~/dts/nn/xgmr/build/debug/osx] $ ls ../output/
ffnn_dynamic.bc      ffnn_dynamic.o       ffnn_dynamic.weights ffnn_static.o        ffnn_static.weights

(base) wyessen [~/dts/nn/xgmr/build/debug/osx] $ 
```

You can safely ignore the generated warning if you are compiling on OSX. 

#### Sample Input

Let us generate sample input for our inference engine. We'll provide two inputs (four floats in a single binary file, two per input, since input dimension is `<1, 2>`). Generate it like this:
```
(base) wyessen [~/dts/nn/xgmr/build/debug/output] $ perl -e 'print pack("f*", 0.123, 0.321, 0.456, 0.654)' > sample_input.dat

(base) wyessen [~/dts/nn/xgmr/build/debug/output] $ od -f sample_input.dat 

0000000     1.230000e-01    3.210000e-01    4.560000e-01    6.540000e-01
0000020

(base) wyessen [~/dts/nn/xgmr/build/debug/output] $ 
```

#### Running with Statically-Linked Bundles

Now with the bundles produced, let us go ahead and compile the inference engine. In this case, we are going to specify the bundle at compile time, statically linking it in. We won't statically link all the other required libraries (e.g. the math library or the GNU argp library). 

The command line to prepare the build system is as follows (sample output is also included).
```
(base) wyessen [~/dts/nn/xgmr/inference_engines/x-inference-engines/build/native/debug] $ cmake -G Ninja -DLIB_ONLY=OFF -DBUILD_FOR_DYNAMIC_LINKAGE=OFF -DLINK_LIBS_STATICALLY=OFF -DENABLE_PERF_MONITORING=OFF -DLINKED_BUNDLE=../../../../../build/debug/output/ffnn_static.o -DLINKED_MODEL_NAME=ffnn ../../../ -DTARGET_OPTIONS=-largp

-- The C compiler identification is AppleClang 10.0.1.10010046
-- The CXX compiler identification is AppleClang 10.0.1.10010046
-- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc
-- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++
-- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
Will build the library and the executable
-- Configuring done
-- Generating done
-- Build files have been written to: /Users/wyessen/dts/nn/xgmr/inference_engines/x-inference-engines/build/native/debug

(base) wyessen [~/dts/nn/xgmr/inference_engines/x-inference-engines/build/native/debug] $ 
```

Note that since we're building on OSX, we had to explicitly specify `-largp` (after installing it with Brew using `brew install argp-standalone`). 

We can now build with `ninja all`:
```
(base) wyessen [~/dts/nn/xgmr/inference_engines/x-inference-engines/build/native/debug] $ ninja all

[1/5] Building C object CMakeFiles/xinfer.dir/x_perf_monitor.c.oclang: warning: -largp: 'linker' input unused [-Wunused-command-line-argument]
[2/5] Building C object CMakeFiles/xinfer.dir/x_inference_lib.c.o
clang: warning: -largp: 'linker' input unused [-Wunused-command-line-argument]
[3/5] Building C object CMakeFiles/x-infer.dir/main.c.o
clang: warning: -largp: 'linker' input unused [-Wunused-command-line-argument]
[4/5] Linking C static library bin/libxinfer.a
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/ranlib: file: bin/libxinfer.a(x_perf_monitor.c.o) has no symbols
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/ranlib: file: bin/libxinfer.a(x_perf_monitor.c.o) has no symbols
[5/5] Linking C executable bin/x-infer

(base) wyessen [~/dts/nn/xgmr/inference_engines/x-inference-engines/build/native/debug] $ 
```

You can safely ignore the generated warnings. At this point the inference engine has been built. You can verify this as follows.

```
(base) wyessen [~/dts/nn/xgmr/inference_engines/x-inference-engines/build/native/debug] $ ./bin/x-infer --help

Usage: x-infer [OPTION...] [WEIGHTS FILENAME]

                    Generic Inference Engine                         
-----------------------------------------------------------------------
Dynamic bundle loading: NOT SUPPORTED (bundle has been statically linked)
Performance monitoring: NOT SUPPORTED

x-infer runs inference against the provided Glow bundle. Weights file must be
specified as the first argument. The input file must be specified with
[--infile] (binary file). Input tensor type [-intype], output tensor type
[-outtype], input length [--inlen], output length [--outlen], input tensor name
[-inname], and output tensor name [--outname] must be specified. 

When built with dynamic bundle loading support, bundle must be specified as the
first positional argument, and the weights file as the second. When built with
a bundle statically linked in, dynamic loading is not supported 

Short and long form options are: 

  -i, --infile=FILE          Input from FILE
  -l, --inlen=LEN            Input tensor length (e.g. if the tensor is of
                             shape 2x3x4, its length is 2 * 3 * 4 = 24)
  -L, --outlen=LEN           Output tensor length (e.g. if the tensor is of
                             shape 2x3x4, its length is 2 * 3 * 4 = 24)
  -n, --inname=NAME          Input tensor name NAME
  -N, --outname=NAME         Output tensor name NAME
  -o, --output=FILE          Output to binary FILE instead of standard output
  -t, --intype=TYPE          Input TYPE (one of F32, F16, I16, I8)
  -T, --outtype=TYPE         Output TYPE (one of F32, F16, I16, I8)
  -?, --help                 Give this help list
      --usage                Give a short usage message
  -V, --version              Print program version

Mandatory or optional arguments to long options are also mandatory or optional
for any corresponding short options.

Report bugs to Github Pytorch Glow repository at
https://github.com/pytorch/glow.

(base) wyessen [~/dts/nn/xgmr/inference_engines/x-inference-engines/build/native/debug] $ 
```

Now let us run on our sample input, saving the output in `output.dat`, and then using `od` to interpret the output.
```
(base) wyessen [~/dts/nn/xgmr/inference_engines/x-inference-engines/build/native/debug] $ ./bin/x-infer --infile=../../../../../build/debug/output/sample_input.dat --inlen=2 --outlen=1 --inname=in --outname=save_out --output=./output.dat --intype=F32 --outtype=F32 ../../../../../build/debug/output/ffnn_static.weights 

(base) wyessen [~/dts/nn/xgmr/inference_engines/x-inference-engines/build/native/debug] $ od -f output.dat 

0000000     5.850238e-01    9.525478e-01                                
0000010

(base) wyessen [~/dts/nn/xgmr/inference_engines/x-inference-engines/build/native/debug] $ 
```

Note that we used the output name `save_out` instead of `out`. The reason for this is because Glow, upon bundle compilation, adds the prefix `save_` to the output tensor name.

#### Running with Dynamically-Loadable Bundles

First, let us recompile the inference engine with support for dynamic bundle loading:
```
(base) wyessen [~/dts/nn/xgmr/inference_engines/x-inference-engines/build/native/debug] $ cmake -G Ninja -DLIB_ONLY=OFF -DBUILD_FOR_DYNAMIC_LINKAGE=ON -DLINK_LIBS_STATICALLY=OFF -DENABLE_PERF_MONITORING=OFF ../../../ -DTARGET_OPTIONS=-largp

-- The C compiler identification is AppleClang 10.0.1.10010046
-- The CXX compiler identification is AppleClang 10.0.1.10010046
-- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc
-- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++
-- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
Will build the library and the executable
Will build the executable for dynamic bundle linkage
-- Configuring done
-- Generating done
-- Build files have been written to: /Users/wyessen/dts/nn/xgmr/inference_engines/x-inference-engines/build/native/debug

(base) wyessen [~/dts/nn/xgmr/inference_engines/x-inference-engines/build/native/debug] $ 
```

Followed by

```
(base) wyessen [~/dts/nn/xgmr/inference_engines/x-inference-engines/build/native/debug] $ ninja all

[1/5] Building C object CMakeFiles/xinfer.dir/x_perf_monitor.c.o
clang: warning: -largp: 'linker' input unused [-Wunused-command-line-argument]
[2/5] Building C object CMakeFiles/xinfer.dir/x_inference_lib.c.o
clang: warning: -largp: 'linker' input unused [-Wunused-command-line-argument]
[3/5] Building C object CMakeFiles/x-infer.dir/main.c.o
clang: warning: -largp: 'linker' input unused [-Wunused-command-line-argument]
[4/5] Linking C static library bin/libxinfer.a
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/ranlib: file: bin/libxinfer.a(x_perf_monitor.c.o) has no symbols
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/ranlib: file: bin/libxinfer.a(x_perf_monitor.c.o) has no symbols
[5/5] Linking C executable bin/x-infer

(base) wyessen [~/dts/nn/xgmr/inference_engines/x-inference-engines/build/native/debug] $ 
```

Again, ignore the generated warnings. You can confirm that dynamic linkage is now supported:

```
(base) wyessen [~/dts/nn/xgmr/inference_engines/x-inference-engines/build/native/debug] $ ./bin/x-infer --help

Usage: x-infer [OPTION...] [BUNDLE FILENAME] [WEIGHTS FILENAME]

                    Generic Inference Engine                         
-----------------------------------------------------------------------
Dynamic bundle loading: SUPPORTED
Performance monitoring: NOT SUPPORTED

x-infer runs inference against the provided Glow bundle. Weights file must be
specified as the first argument. The input file must be specified with
[--infile] (binary file). Input tensor type [-intype], output tensor type
[-outtype], input length [--inlen], output length [--outlen], input tensor name
[-inname], and output tensor name [--outname] must be specified. 

When built with dynamic bundle loading support, bundle must be specified as the
first positional argument, and the weights file as the second. When built with
a bundle statically linked in, dynamic loading is not supported 

Short and long form options are: 

  -i, --infile=FILE          Input from FILE
  -l, --inlen=LEN            Input tensor length (e.g. if the tensor is of
                             shape 2x3x4, its length is 2 * 3 * 4 = 24)
  -L, --outlen=LEN           Output tensor length (e.g. if the tensor is of
                             shape 2x3x4, its length is 2 * 3 * 4 = 24)
  -m, --model=NAME           Model name (maximum 128 chars)
  -n, --inname=NAME          Input tensor name NAME
  -N, --outname=NAME         Output tensor name NAME
  -o, --output=FILE          Output to binary FILE instead of standard output
  -t, --intype=TYPE          Input TYPE (one of F32, F16, I16, I8)
  -T, --outtype=TYPE         Output TYPE (one of F32, F16, I16, I8)
  -?, --help                 Give this help list
      --usage                Give a short usage message
  -V, --version              Print program version

Mandatory or optional arguments to long options are also mandatory or optional
for any corresponding short options.

Report bugs to Github Pytorch Glow repository at
https://github.com/pytorch/glow.

(base) wyessen [~/dts/nn/xgmr/inference_engines/x-inference-engines/build/native/debug] $ 
```

We can now run the inference engine, specifying the bundle we wish to use as the first positional argument, as well as the `--model` option specifying the model name:

```
(base) wyessen [~/dts/nn/xgmr/inference_engines/x-inference-engines/build/native/debug] $ ./bin/x-infer --infile=../../../../../build/debug/output/sample_input.dat --inlen=2 --outlen=1 --inname=in --outname=save_out --output=./output.dat --intype=F32 --outtype=F32 ../../../../../build/debug/output/ffnn_dynamic.o ../../../../../build/debug/output/ffnn_dynamic.weights --model=ffnn

(base) wyessen [~/dts/nn/xgmr/inference_engines/x-inference-engines/build/native/debug] $ od -f output.dat 
0000000     5.850238e-01    9.525478e-01                                
0000010

(base) wyessen [~/dts/nn/xgmr/inference_engines/x-inference-engines/build/native/debug] $ 
```

#### Performance Monitoring (Linux Only)

We can compile with the `-DENABLE_PERF_MONITORING` option (Linux only) to enable performance monitoring. After this, performance will be reported as in the following example. Note that not all VMs support this, and when running in a VM you may not get sensible results. Also, when running natively, it is often the case that you'll need to run with `sudo` to avoid permission errors (due to how performance metrics are gathered, elevated privileges may be required).

```
wyessen@raspberrypi:~/workbench/glow $ sudo ./bin/x-infer ./bundles/00_ffnn_nonquantized/ffnn.o ./bundles/00_ffnn_nonquantized/ffnn.weights -i ./data/00_ffnn_nonquantized/00_dummy_input.dat -l 2 -L 1 -m ffnn -n in -N save_out -o output.dat -p -t F32 -T F32

Constant weights size       : 896 bytes
Number of cases             : 1
Number of CPU cycles (x1-e6): 0.014533
```
Note that CPU cycles are counted in millions.

### Technical Notes and Notes on Dependencies

1. Currently the engine is written in pure C (C99) and as such can be built and linked with pure C tools.
2. As mentioned above, only UNIX-like platforms are currently supported (and for some features, only Linux). In addition,
   GNU argument parsing library (GNU Argp) is required when building the stand-alone application. Please consult the appropriate documentation for your platform for help with installation. It is provided by default with the GNU build tools on many UNIX(-like) platforms, including Linux, and can be installed with `brew install argp-standalone` on OSX. When compiling on OSX, the additional `-DTARGET_OPTIONS=-largp` must be specified.
3. Streaming is currently not supported, but this feature is planned.
4. More than one named input, or more than one named output, is currently not supported. This support is planned in the future.
5. Error reporting is rather rudimentary. More detailed error messages are needed. This is planned for the future.
6. It may be possible to figure out input and output tensor lengths from the bundle, so that the `inlen` and `outlen` options can be safely dropped. This is planned for the future. Similarly for input and output types (although this would be more challenging).

### A Note on the Initial Contribution

The initial contribution of the `x-inference-engine` (as well as the corresponding documentation) was made as part of the open source contribution initiative by the XPERI Corporation (xperi.com). 