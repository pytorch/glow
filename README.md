# Noether

Noether is a machine learning compiler and inference engine for hardware
accelerators. This library is designed to be used as a backend for some machine
learning framework. The compiler is designed to allow state of the art compiler
optimizations on neural network graphs.

## Getting Started

### System Requirements

The project builds and runs on macOS and Linux. The software depends on a modern
C++ compiler that supports C++11, on CMake, protocol buffer, and libpng.

### Building the Compiler

Create a build directory and run cmake on the source directory:

  mkdir build; cmake ../noether/

It is possible to configure and build the compiler with any CMake generator,
like GNU Makefiles, Ninja and Xcode build.

## Testing and Running

After compiling the project, a few test programs will be built under the
/examples/ directory. The 'mnist' and 'cifar10' programs train and run the digit
recognition and image classification benchmarks. The programs 'tensors' and
'regress' are small programs that run a few tests that check different parts of
the runtime.

The default compilation mode is 'Debug'. This means that the compiler itself is
easy to debug because it has debug info, lots of assertions, and the
optimizations are disabled. It also means that the compiler and runtime are very
slow, and the execution time can be hundreds of times slower than that of
release builds.  If you wish to benchmark the compiler or release it then you
should compile the compiler in Release mode. Check the main CMake file for more
details.

