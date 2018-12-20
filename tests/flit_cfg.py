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

# Load a pre-trained Caffe2 image classifier and run it on an image.

# Configuration file for the 'lit' test runner.

import os

def getToolPath(tool_name, tools_dir):
    tool_path = os.path.normpath(os.path.join(tools_dir, tool_name))
    if not os.path.exists(tool_path):
        return None
    return tool_path

def setup_config(config):
    # name: The name of this test suite.
    config.name = 'glow'

    # suffixes: A list of file extensions to treat as test files.
    config.suffixes = [".test"]

    # Set the different string substitutions that we use in our test suite.
    config.tool_to_path = {}

    for tool_name in ('text-translator', 'image-classifier', 'model-runner'):
        tool_exe = getToolPath(tool_name, config.tools_dir)
        config.tool_to_path[tool_name] = tool_exe
        if not tool_exe:
            # Skip the substitution for tools that are not here.
            # We need to skip those otherwise lit will choke when
            # registering the substitutions because it doesn't expect
            # empty substitution.
            continue
        # Use insert instead of append, because some standard substitutions are
        # substring of our substitutions and would kick before ours.
        # E.g., %t (tmpFile expansion) would be applied before %text-translator,
        # leaving us with <tmpFile>ext-translator and that's not what we want.
        config.substitutions.insert(0, ('%' + tool_name, tool_exe))

    if config.release_mode:
        config.available_features.add('release')
    if config.has_opencl:
        config.available_features.add('opencl')
    if config.has_cpu:
        config.available_features.add('cpu')
    if config.models_dir:
        config.substitutions.insert(0, ('%models_dir', config.models_dir))
        if os.path.isdir(config.models_dir):
            config.available_features.add('models')

    config.substitutions.insert(0, ('%FileCheck', config.filechecker_command))

    # test_source_root: The root path where tests are located.
    config.test_source_root = os.path.dirname(__file__)

    # test_exec_root: The root path where tests should be run.
    config.test_exec_root = os.path.join(config.binary_dir, 'tests')
