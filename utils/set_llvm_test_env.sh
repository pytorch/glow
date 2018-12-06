#!/usr/bin/env bash

# Copyright (c) 2018-present, Facebook, Inc.
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

LLVM_DIR="$PWD/llvm_src"

if [ ! -e "$LLVM_DIR" ]; then
    git clone --depth=1 -b release_60 https://github.com/llvm-mirror/llvm.git llvm_src
fi

# Check if lit is already available.
# If not, add it in the PATH.
if ! which lit.py > /dev/null; then
    export PATH="$PATH:$LLVM_DIR/utils/lit"
fi

# Check if FileCheck is already available.
# If not, build it.
if ! which FileCheck > /dev/null; then
    FILECHECK_BUILDDIR="$LLVM_DIR/build_filecheck"
    mkdir $FILECHECK_BUILDDIR
    (
        cd $FILECHECK_BUILDDIR
        cmake $LLVM_DIR -GNinja -DCMAKE_BUILD_TYPE=Release
        ninja FileCheck
    )
    export PATH="$PATH:$FILECHECK_BUILDDIR/bin"
    # If we cannot get FileCheck, there is no point in this
    # build variant.
    if ! which FileCheck > /dev/null; then
        echo "Cannot find FileCheck."
        exit -1
    fi
fi
