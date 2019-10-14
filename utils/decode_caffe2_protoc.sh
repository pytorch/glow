#!/usr/bin/env bash

# Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

# Decodes a binary pb file of a Caffe2 model to textual pbtxt format.
if [ -z "$1" ]; then
    echo "Usage: $(basename "$0") pb"
    exit 1
fi

PB="$1"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

LIB_DIR="$SCRIPT_DIR/../lib/Importer"

protoc --decode=caffe2.NetDef -I "$LIB_DIR" "$LIB_DIR/caffe2.proto" < "$PB"
