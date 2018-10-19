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

if [ ! -e "./llvm_src/" ]; then
  git clone --depth=1 -b release_60 https://github.com/llvm-mirror/llvm.git llvm_src
fi

mkdir -p llvm_build
mkdir -p llvm_install

BASE=$PWD

cd llvm_build
cmake ../llvm_src/ -G Ninja -DCMAKE_INSTALL_PREFIX="$BASE/llvm_install" -DCMAKE_BUILD_TYPE=Release
cmake --build . --target install

echo "Built LLVM into " "$BASE/llvm_install"
