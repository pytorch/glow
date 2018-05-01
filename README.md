# Glow

* [![Build Status](https://travis-ci.com/pytorch/Glow.svg?token=UwQBGB2pxogBqjigi7Nh&branch=master)](https://travis-ci.com/pytorch/Glow)
* [Code Coverage](https://fb-glow-assets.s3.amazonaws.com/coverage/coverage-master/index.html)

Glow is a machine learning compiler and execution engine for hardware
accelerators. This library is designed to be used as a backend for the Caffe2
machine learning framework. The compiler is designed to allow state of the art
compiler optimizations and code generation of neural network graphs.  This
library is experimental and in active development.

## How does it work?

The Glow compiler has three different
[intermediate representations](./docs/IR.md) at different phases of the
compilation pipe. The first representation is a high-level graph that resembles
the original neural network. This representation allows the compiler to perform
high-level domain-specific optimizations. The next level is a low-level
bytecode, that represents memory explicitly and allows the compiler to perform
low-level memory optimizations that are not possible at higher levels. And
finally, the target-specific intermediate representation that the code
generators can use to generate efficient machine code.

![](./docs/3LevelIR.png)

## Getting Started

### System Requirements

Glow builds and runs on macOS and Linux. The software depends on a modern C++
compiler that supports C++11, on CMake, LLVM, protocol buffers, and libpng.

#### Submodules

Glow depends on googletest as a submodule.  To get it, run:

  ```
  git submodule update --init --recursive
  ```

#### macOS

Install the required dependencies using [Homebrew](https://brew.sh/):

  ```
  brew install cmake graphviz libpng ninja protobuf wget
  brew install --with-toolchain llvm
  ```

Note that LLVM is installed to a non-default location (`/usr/local/opt/llvm`) to
avoid conflicts with the system's LLVM.

#### Ubuntu

On Ubuntu you would need to install a few dependencies. The following command
should install the required dependencies.

  ```
  sudo apt-get install graphviz clang cmake wget ninja-build llvm-5.0 \
      libprotobuf-dev protobuf-compiler
  ```

In order to support ONNX net serialization format, Glow requires
`protobuf >= 2.6.1`, but the above command may install older
version on older Ubuntu (e.g. 14.04). If this is the case, we suggest to look
at `utils/install_protobuf.sh` to install newer version from source.

Note, that OpenCL support is not trivial on Linux. We suggest to build without
OpenCL for the first time.

### Configure and build

To build the compiler, create a build directory and run cmake on the source
directory. It's a good idea to build two configurations (Release and Debug)
because some programs take a really long time to run in Debug mode. It's also a
good idea to build the project outside of the source directory.

  ```
  mkdir build_Debug
  cd build_Debug
  cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug ../Glow
  ninja all
  ```

It's possible to configure and build the compiler with any CMake generator,
like GNU Makefiles, Ninja and Xcode build.

### Building with dependencies (LLVM)

By default, Glow will use a system provided LLVM.  Note that Glow requires LLVM
5.0 or later.  If you have LLVM installed in a non-default location (for
example, if you installed it using Homebrew on macOS), you need to tell CMake
where to find llvm using `-DCMAKE_PREFIX_PATH`.  For example:

  ```
  cmake -G Ninja ../Glow \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_PREFIX_PATH=/usr/local/opt/llvm
  ```

If LLVM is not available on your system you'll need to build it manually.  Run
the script '`/utils/build_llvm.sh` to clone, build and install LLVM in a local
directory. You will need to configure Glow with the flag `-DCMAKE_PREFIX_PATH`
to tell the build system where to find LLVM (e.g. the location of
`llvm_install/` if using `build_llvm.sh`).

For more platform-specific build instructions and advanced options, such as
building with Address-Sanitizers refer to this guide:
[Building the Compiler](docs/Building.md).

## Testing and Running

### Unit tests

The project has a few unit tests in the tests/unittests subdirectory. To run all
of them, simply run `ninja test`.

### C++ API examples

A few test programs that use Glow's C++ API are found under the `examples/`
subdirectory. The `mnist`, `cifar10`, `fr2en` and `ptb` programs train and run digit
recognition, image classification and language modeling benchmarks,
respectively.

To run these programs, build Glow in Release mode, then run the following commands
to download the cifar10, mnist and ptb databases.

  ```
  python ../Glow/utils/download_test_db.py --all
  ```

Now run the examples. Note that the databases should be in the current working
directory.

  ```
  ./bin/mnist
  ./bin/cifar10
  ./bin/fr2en
  ./bin/ptb
  ```

If everything goes well you should see:
  * `mnist`: pictures from the mnist digits database
  * `cifar10`: image classifications that steadily improve
  * `fr2en`: an interactive French-to-English translator
  * `ptb` decreasing perplexity on the dataset as the network trains

Note that the default build mode is `Debug`, which means that the compiler
itself is easy to debug because the binary contains debug info, lots of
assertions, and the optimizations are disabled. It also means that the compiler
and runtime are very slow, and the execution time can be hundreds of times
slower than that of release builds. If you wish to benchmark the compiler, run
long benchmarks, or release the product then you should compile the compiler in
Release mode. Check the main CMake file for more details.

More details on testing and running Glow can be found in: [Testing the Glow
Compiler](docs/Testing.md).

### Ahead-of-time Compilation

Glow can be used to compile neural networks into object files containing native
code.  We provide resnet50 (both quantized and non-quantized versions) as an
example of this capability in `examples/bundles/resnet50`.  See [Creating
Standalone Executable Bundles](docs/AOT.md) for more detail.

## Contributing

To get started, please refer to the following guides:
* [Contributing](CONTRIBUTING.md)
* [CodingStandards](docs/CodingStandards.md)
