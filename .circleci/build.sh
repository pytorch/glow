#!/bin/bash

# This script setup an working environemnt for running glow tests and gtest driver.
# By default, we run with the enabled CPU backend and disabled OpenCL backend.
set -ex

# Add support for https apt sources.
wget http://security.ubuntu.com/ubuntu/pool/main/a/apt/apt-transport-https_1.2.32ubuntu0.2_amd64.deb
echo "93475e4cc5e7a86de63fea0316f3f2cd8b791cf4d6ea50a6d63f5bd8e1da5726  apt-transport-https_1.2.32ubuntu0.2_amd64.deb" | sha256sum -c
sudo dpkg -i apt-transport-https_1.2.32ubuntu0.2_amd64.deb
rm apt-transport-https_1.2.32ubuntu0.2_amd64.deb

export MAX_JOBS=8
if [ "${CIRCLE_JOB}" != "COVERAGE" ]; then
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
fi

install_pocl() {
   sudo apt-get install -y ocl-icd-opencl-dev clinfo libhwloc-dev opencl-headers

   git clone https://github.com/pocl/pocl.git
   cd pocl && git checkout 4efafa82c087b5e846a9f8083d46b3cdac2f698b && cd ../
   mkdir build_pocl
   cd build_pocl
   cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/clang++-8 -DCMAKE_C_COMPILER=/usr/bin/clang-8 -DENABLE_ICD=ON ../pocl
   make -j`nproc`
   sudo make install

   sudo mkdir -p /etc/OpenCL/vendors/
   sudo cp /usr/local/etc/OpenCL/vendors/pocl.icd /etc/OpenCL/vendors/

   clinfo
   cd ../
}

install_fmt() {
    git clone https://github.com/fmtlib/fmt --branch 7.1.3
    pushd fmt
    mkdir build
    cd build
    cmake ..
    make -j`nproc`
    sudo make install
    popd
}

upgrade_python() {
    echo "Removing old python...";
    sudo apt-get remove --purge -y python3.6 python3-pip libpython3-dev
    sudo apt-get autoremove -y

    echo "Installing dependencies for new python..."
    sudo apt-get update
    sudo apt-get install -y libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev

    echo "Installing new python..."
    mkdir python-src
    pushd python-src
    wget https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz
    tar xvf Python-3.9.0.tgz
    cd Python-3.9.0
    ./configure --enable-shared
    sudo make altinstall
    popd

    echo "Adjusting system to recognize new python..."
    sudo touch /etc/ld.so.conf.d/glowDevLibs.conf
    echo "/usr/local/lib/" | sudo tee -a /etc/ld.so.conf.d/glowDevLibs.conf
    sudo ldconfig
    sudo rm /usr/local/bin/pip
    sudo ln -s /usr/local/bin/pip3.9 /usr/local/bin/pip

    echo "Installing virtualenv..."
    sudo pip3.9 install virtualenv
}

GLOW_DEPS="libpng-dev libgoogle-glog-dev libboost-all-dev libdouble-conversion-dev libgflags-dev libjemalloc-dev libpthread-stubs0-dev libevent-dev libssl-dev"

if [ "${CIRCLE_JOB}" == "CHECK_CLANG_AND_PEP8_FORMAT" ]; then
    sudo apt-get update
    upgrade_python
else
    # Install Glow dependencies
    sudo apt-get update

    # Redirect clang
    sudo ln -s /usr/bin/clang-8 /usr/bin/clang
    sudo ln -s /usr/bin/clang++-8 /usr/bin/clang++
    sudo ln -s /usr/bin/llvm-symbolizer-8 /usr/bin/llvm-symbolizer
    sudo ln -s /usr/bin/llvm-config-8 /usr/bin/llvm-config-8.0

    sudo apt-get install -y ${GLOW_DEPS}
    install_fmt
fi

# Since we are using llvm-7 in these two branches, we cannot use pip install cmake
if [ "${CIRCLE_JOB}" != "PYTORCH" ] && [ "${CIRCLE_JOB}" != "CHECK_CLANG_AND_PEP8_FORMAT" ]; then
    sudo pip install cmake==3.17.3
else
    sudo apt-get install cmake
fi

# Install ninja, (newest version of) autopep8 through pip
sudo pip install ninja
hash cmake ninja

# Build glow
GLOW_DIR=$PWD
cd ${GLOW_DIR}
mkdir build && cd build

CMAKE_ARGS=()

CMAKE_ARGS+=("-DCMAKE_CXX_FLAGS=-Werror")
CMAKE_ARGS+=("-DGLOW_WITH_CPU=ON")
CMAKE_ARGS+=("-DGLOW_WITH_HABANA=OFF")

if [[ "${CIRCLE_JOB}" == "ASAN" ]]; then
    CMAKE_ARGS+=("-DGLOW_USE_SANITIZER='Address;Undefined'")
    CMAKE_ARGS+=("-DGLOW_WITH_OPENCL=OFF")
    CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Release")
elif [[ "$CIRCLE_JOB" == "COVERAGE" ]]; then
    sudo apt-get install wget
    sudo apt-get install -y lcov
    sudo pip install awscli --upgrade
    ../utils/install_protobuf.sh
    CC=gcc-5 CXX=g++-5 cmake -G Ninja \
          -DCMAKE_BUILD_TYPE=Debug -DGLOW_WITH_OPENCL=OFF -DGLOW_WITH_CPU=ON \
          -DLLVM_DIR=/usr/lib/llvm-7/cmake \
          -DGLOW_USE_COVERAGE=ON \
          ../
elif [[ "$CIRCLE_JOB" == "CHECK_CLANG_AND_PEP8_FORMAT" ]]; then
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
    sudo apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-11 main"
    sudo apt-get update
    sudo apt-get install -y clang-format-11
    cd /tmp
    python3.9 -m virtualenv venv
    source venv/bin/activate
    pip install black==22.3.0
    cd ${GLOW_DIR}
elif [[ "$CIRCLE_JOB" == "OPENCL" ]]; then
    install_pocl
    CMAKE_ARGS+=("-DGLOW_WITH_OPENCL=ON")
elif [[ "$CIRCLE_JOB" == "FEATURE_COMPILATION" ]]; then
    CMAKE_ARGS+=("-DGLOW_USE_PNG_IF_REQUIRED=OFF")
elif [[ "$CIRCLE_JOB" == "32B_DIM_T" ]]; then
    install_pocl
    CMAKE_ARGS+=("-DTENSOR_DIMS_32_BITS=ON -DGLOW_WITH_OPENCL=ON")
else
    CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Debug")
    if [[ "${CIRCLE_JOB}" == "SHARED" ]]; then
        CMAKE_ARGS+=("-DBUILD_SHARED_LIBS=ON")
    fi
fi

if [ "${CIRCLE_JOB}" != "COVERAGE" ] && [ "${CIRCLE_JOB}" != "CHECK_CLANG_AND_PEP8_FORMAT" ] && [ "${CIRCLE_JOB}" != "PYTORCH" ]; then
    cmake -GNinja ${CMAKE_ARGS[*]} ../
    ninja
fi

# Report sccache hit/miss stats
if hash sccache 2>/dev/null; then
    sccache --show-stats
fi
