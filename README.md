# Glow

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
take a really long time to run in Debug mode. It is also a good idea to build
the project outside of the source directory.

  ```
  mkdir build
  cd build
  cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug ../glow
  ```

It is possible to configure and build the compiler with any CMake generator,
like GNU Makefiles, Ninja and Xcode build.

### Building with the Sanitizers

Google's sanitizer project provides a number of libraries which can be used with
compiler inserted instrumentation to find a variety of bugs at runtime.  These
include memory issues due such as use-after-free or double-free.  They can also
detect other types of problems like memory leaks.  Glow can be built with the
sanitizers enabled using an additional parameter to cmake.

The following sanitizers are currently configurable:

  - Address
  - Memory
  - Undefined
  - Thread
  - Leaks

You can pass one of the above as a value to the cmake parameter
`GLOW_USE_SANITIZER`.  `Address` and `Undefined` are special in that they may be
enabled simultaneously by passing `Address;Undefined` as the value.
Additionally, the `Memory` sanitizer can also track the origin of the memory
which can be enabled by using `MemoryWithOrigins` instead of `Memory`.

### Building with clang-tidy

The clang project provides an additional utility to scan your source code for
possible issues.  Enabling `clang-tidy` checks on the source code is easy and
can be done by passing an additional cmake parameter during the configure step.

  ```
  -DCMAKE_CXX_CLANG_TIDY=$(which clang-tidy)
  ```

Adding this to the configure step will automatically run clang-tidy on the
source tree during compilation. Use the following configuration to enable
auto-fix and to enable/disable specific checks:

  ```
  -DCMAKE_CXX_CLANG_TIDY:STRING="$(which clang-tidy);-checks=...;-fix"
  ```

## Testing and Running

The project has a few unittests in the tests/ directory. To run all of the unit
tests simply run the command 'ninja test' (or gmake test).  After compiling the
project, a few test programs will be built under the /examples/ directory. The
'mnist' and 'cifar10' programs train and run the digit recognition and image
classification benchmarks.

The default compilation mode is 'Debug'. This means that the compiler itself is
easy to debug because the binary contains debug info, lots of assertions, and
the optimizations are disabled. It also means that the compiler and runtime are
very slow, and the execution time can be hundreds of times slower than that of
release builds. If you wish to benchmark the compiler, run long benchmarks, or
release the product then you should compile the compiler in Release mode. Check
the main CMake file for more details.

After building Glow in Release-mode run the following command to download the
cifar10 and mnist database:

```
python ../glow/utils/download_test_db.py
```

Next, after downloading and extracting the mnist and cifar10 database
(preferably in the build directory), you can run the test programs:

```
./bin/mnist
./bin/cifar10
```

Note: The databases should be (for now) in the same directory from where the
executable is run.

If everything goes well you should see pictures from the mnist digits database
and print outs from cifar10 that make sense.

