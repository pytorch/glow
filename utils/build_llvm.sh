#!/usr/bin/env bash

set -euo pipefail

if [ ! -e "./llvm_src/" ]; then
  git clone --depth=1 -b release_60 http://llvm.org/git/llvm.git llvm_src
fi

mkdir -p llvm_build
mkdir -p llvm_install

BASE=$PWD

cd llvm_build
cmake ../llvm_src/ -G Ninja -DCMAKE_INSTALL_PREFIX="$BASE/llvm_install" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build . --target install

echo "Built LLVM into " "$BASE/llvm_install"
