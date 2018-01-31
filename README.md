# Glow

[![Build Status](https://travis-ci.com/facebookexternal/Glow.svg?token=UwQBGB2pxogBqjigi7Nh&branch=master)](https://travis-ci.com/facebookexternal/Glow)

Glow is a machine learning compiler and inference engine for hardware
accelerators. This library is designed to be used as a backend for some machine
learning framework. The compiler is designed to allow state of the art compiler
optimizations on neural network graphs.

## Getting Started

### System Requirements

The project builds and runs on macOS and Linux. The software depends on a modern
C++ compiler that supports C++11, on CMake, protocol buffer, and libpng.

### Building the Compiler

Next, create a build directory and run cmake on the source directory. It is a
good idea to build two configurations (Release and Debug) because some programs
take a really long time to run in Debug mode. It's also a good idea to build
the project outside of the source directory.

  ```
  mkdir build_Debug
  cd build_Debug
  cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug ../glow
  ninja all
  ```

It's possible to configure and build the compiler with any CMake generator,
like GNU Makefiles, Ninja and Xcode build.

### Building with dependencies (LLVM)

By default, Glow will use a system provided LLVM.  Note that Glow requires LLVM
5.0. If LLVM is not available on your system you'll need to build it manually.
You may find the script './utils/build\_llvm.sh"' useful. You will need to
configure Glow with the flag '-DCMAKE\_PREFIX\_PATH' to tell the build system
where to find LLVM.

For more platform-specific build instructions and advance options, such as
building with Address-Sanitizers refer to this guide:
[Contributing](docs/Building.md)


## Testing and Running

The project has a few unittests in the tests/ directory. To run all of the unit
tests simply run the command 'ninja test' (or gmake test).  After compiling the
project, a few test programs will be built under the /examples/ directory. The
'mnist', 'cifar10' and 'ptb' programs train and run the digit recognition, image
classification and language modeling benchmarks.

The default compilation mode is 'Debug'. This means that the compiler itself is
easy to debug because the binary contains debug info, lots of assertions, and
the optimizations are disabled. It also means that the compiler and runtime are
very slow, and the execution time can be hundreds of times slower than that of
release builds. If you wish to benchmark the compiler, run long benchmarks, or
release the product then you should compile the compiler in Release mode. Check
the main CMake file for more details.

After building Glow in Release-mode run the following command to download the
cifar10, mnist and ptb database:

```
python ../glow/utils/download_test_db.py --all
```

Next, after downloading and extracting the mnist and cifar10 database
(preferably in the build directory), you can run the test programs:

```
./bin/mnist
./bin/cifar10
./bin/ptb
```

Note: The databases should be (for now) in the same directory from where the
executable is run.

If everything goes well you should see pictures from the mnist digits database
and print outs from cifar10 that make sense as well as the perplexity on the
ptb dataset go down as the network trains.

## Contributing

To get started please refer to the following guides:
* [Contributing](docs/Contributing.md)
* [CodingStandards](docs/CodingStandards.md)

