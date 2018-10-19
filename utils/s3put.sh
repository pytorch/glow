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

if [ -z "$1" ]; then
    echo "Usage: $(basename "$0") model-directory"
    exit 1
fi

MODEL=$1

# You'll need an access key and secret key for this to work.  Search the
# FB-internal Glow group for `s3cmd` to find it.
s3cmd put --recursive --acl-public "$MODEL" s3://fb-glow-assets/models/
