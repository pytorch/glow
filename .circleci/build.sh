#!/bin/bash

# This script setup an working environemnt for running glow tests and gtest driver.
# By default, we run with the enabled CPU backend and disabled OpenCL backend.
set -ex

export MAX_JOBS=8

install_pocl() {
   sudo apt-get install -y ocl-icd-opencl-dev clinfo libhwloc-dev
   
   wget https://github.com/pocl/pocl/archive/v1.2.tar.gz
   tar xf v1.2.tar.gz
   mkdir build_pocl
   cd build_pocl
   cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_ICD=ON ../pocl-1.2
   make -j`nproc`
   sudo make install

   sudo mkdir -p /etc/OpenCL/vendors/
   sudo cp /usr/local/etc/OpenCL/vendors/pocl.icd /etc/OpenCL/vendors/

   clinfo
   cd ../
}

# setup sccache wrappers
if hash sccache 2>/dev/null; then
    SCCACHE_BIN_DIR="/tmp/sccache"
    mkdir -p "$SCCACHE_BIN_DIR"
    for compiler in cc c++ gcc g++ x86_64-linux-gnu-gcc; do
        (
            echo "#!/bin/sh"
            echo "exec $(which sccache) $(which $compiler) \"\$@\""
        ) > "$SCCACHE_BIN_DIR/$compiler"
        chmod +x "$SCCACHE_BIN_DIR/$compiler"
    done
    export PATH="$SCCACHE_BIN_DIR:$PATH"
fi

GLOW_DIR=$PWD

# Install Glow dependencies
sudo apt-get update
sudo apt-get install -y llvm-6.0 llvm-6.0-dev libpng-dev

# Redirect clang
sudo ln -s /usr/bin/clang-6.0 /usr/bin/clang
sudo ln -s /usr/bin/clang++-6.0 /usr/bin/clang++

# Install ninja and (newest version of) cmake through pip
sudo pip install ninja cmake
hash cmake ninja

# Build glow
cd ${GLOW_DIR}
mkdir build && cd build
CMAKE_ARGS=()
CMAKE_ARGS+=("-DGLOW_WITH_CPU=ON")
if [[ "$CIRCLE_JOB" == ASAN ]]; then
    CMAKE_ARGS+=("-DGLOW_USE_SANITIZER='Address;Undefined'")
fi
if [[ "$CIRCLE_JOB" == DEBUG ]]; then
    install_pocl
    CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Debug")
    CMAKE_ARGS+=("-DGLOW_WITH_OPENCL=ON")
    CMAKE_ARGS+=("-DGLOW_RUN_OPENCL_TESTS=OFF")
else
    CMAKE_ARGS+=("-DGLOW_WITH_OPENCL=OFF")
    CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Release")
fi
cmake -GNinja ${CMAKE_ARGS[*]} ../
ninja

# Build onnxifi test driver (Only for DEBUG mode)
if [[ "$CIRCLE_JOB" == DEBUG ]]; then
    ONNX_DIR="${GLOW_DIR}/thirdparty/onnx"
    cd ${ONNX_DIR}
    mkdir build_onnx && cd build_onnx
    cmake -GNinja -DONNX_BUILD_TESTS=ON -DONNXIFI_DUMMY_BACKEND=OFF ../
    ninja onnxifi_test_driver_gtests
    cp ${ONNX_DIR}/build_onnx/onnxifi_test_driver_gtests ${GLOW_DIR}/build
fi

# Report sccache hit/miss stats
if hash sccache 2>/dev/null; then
    sccache --show-stats
fi
