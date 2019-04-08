#!/bin/bash

# This script setup an working environemnt for running glow tests and gtest driver.
# By default, we run with the enabled CPU backend and disabled OpenCL backend.
set -ex

export MAX_JOBS=8

install_pocl() {
   sudo apt-get install -y ocl-icd-opencl-dev clinfo libhwloc-dev

   git clone https://github.com/pocl/pocl.git
   cd pocl && git checkout 94fba9f510e678cd7f8fc988c01618e1ae93dfdf && cd ../
   mkdir build_pocl
   cd build_pocl
   cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_ICD=ON ../pocl
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
sudo ln -s /usr/bin/llvm-symbolizer-6.0 /usr/bin/llvm-symbolizer

# Install ninja and (newest version of) cmake through pip
sudo pip install ninja cmake
hash cmake ninja

# Build glow
cd ${GLOW_DIR}
mkdir build && cd build
CMAKE_ARGS=("-DCMAKE_CXX_FLAGS=-Werror")
CMAKE_ARGS+=("-DGLOW_WITH_CPU=ON")
CMAKE_ARGS+=("-DGLOW_WITH_HABANA=OFF")
if [[ "${CIRCLE_JOB}" == "ASAN" ]]; then
    CMAKE_ARGS+=("-DGLOW_USE_SANITIZER='Address;Undefined'")
    CMAKE_ARGS+=("-DGLOW_WITH_OPENCL=OFF")
    CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Release")
elif [[ "${CIRCLE_JOB}" == "TSAN" ]]; then
    CMAKE_ARGS+=("-DGLOW_USE_SANITIZER='Thread'")
    CMAKE_ARGS+=("-DGLOW_WITH_OPENCL=OFF")
elif [[ "$CIRCLE_JOB" == RELEASE_WITH_EXPENSIVE_TESTS ]]; then
    # Download the models and tell cmake where to find them.
    MODELS_DIR="$GLOW_DIR/downloaded_models"
    DOWNLOAD_EXE="$GLOW_DIR/utils/download_caffe2_models.sh"
    mkdir $MODELS_DIR
    (
        cd $MODELS_DIR
        $DOWNLOAD_EXE
    )
    CMAKE_ARGS+=("-DGLOW_MODELS_DIR=$MODELS_DIR")
    CMAKE_ARGS+=("-DGLOW_WITH_OPENCL=OFF")
    CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Release")
else
    install_pocl
    CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Debug")
    CMAKE_ARGS+=("-DGLOW_WITH_OPENCL=ON")
    if [[ "${CIRCLE_JOB}" == "SHARED" ]]; then
        CMAKE_ARGS+=("-DBUILD_SHARED_LIBS=ON")
    fi
fi

cmake -GNinja ${CMAKE_ARGS[*]} ../
ninja

# Build onnxifi test driver (Only for DEBUG mode)
if [[ "$CIRCLE_JOB" == DEBUG ]]; then
    cd ${GLOW_DIR}
    ./tests/onnxifi/build_onnxifi_tests.sh
fi

# Report sccache hit/miss stats
if hash sccache 2>/dev/null; then
    sccache --show-stats
fi
