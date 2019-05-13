#!/usr/bin/env bash

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -euo pipefail

RELEASE=${RELEASE:-release_80}
if [ ! -e "./llvm/" ]; then
    git clone --depth=1 -b "$RELEASE" https://github.com/llvm-mirror/llvm.git llvm
fi
if [ ! -e "./clang/" ]; then
  git clone --depth=1 -b "$RELEASE" https://github.com/llvm-mirror/clang.git clang
fi

mkdir -p llvm_build
mkdir -p llvm_install

BASE=$PWD

cd llvm_build
# LLVM_INSTALL_UTILS adds the utilities like FileCheck to the install
cmake ../llvm/ -G Ninja -DCMAKE_INSTALL_PREFIX="$BASE/llvm_install" \
      -DCMAKE_BUILD_TYPE=Release -DLLVM_INSTALL_UTILS=ON \
      -DLLVM_ENABLE_PROJECTS=clang
cmake --build . --target install

echo "Built LLVM into " "$BASE/llvm_install"
