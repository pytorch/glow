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

Before building the compiler you will need to update the git submodules with the
command:

  git submodule update

Next, create a build directory and run cmake on the source directory:

  ```
  mkdir build
  cd build
  cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTING=YES ..
  ```

It is possible to configure and build the compiler with any CMake generator,
like GNU Makefiles, Ninja and Xcode build.

### Building with the Sanitizers

Google's santizier project provides a number of libraries which can be used with
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

