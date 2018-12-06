# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os
import sys
import re
import platform
import subprocess
import imp

import lit.util
import lit.formats

# Play with the importer path to import the cmake generated config file.
build_dir = os.getcwdu()
path = os.path.join(build_dir, 'tests', 'litconfig.py')
litconfig = imp.load_source('litconfig', path)

import litconfig

def getToolPath(tool_name, tools_dir):
    tool_exe = lit.util.which(tool_name, tools_dir)
    if not tool_exe:
        return None
    return tool_exe

# name: The name of this test suite.
config.name = 'glow'

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(execute_external=True)

# suffixes: A list of file extensions to treat as test files. This is overriden
# by individual lit.local.cfg files in the test subdirectories.
config.suffixes = ['.test']

# Set the different string substitutions that we use in our test suite.
tools_dir = os.path.join(build_dir, 'bin')
config.tool_to_path = {}

for tool_name in ('text-translator', 'image-classifier', 'model-runner'):
    tool_exe = getToolPath(tool_name, tools_dir)
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


# Set the available feature based on the actual cmake variables.
litconfig.set_glow_available_features(config)

if config.release_mode:
    config.available_features.add('release')
if config.has_opencl:
    config.available_features.add('opencl')
if config.has_cpu:
    config.available_features.add('cpu')
if config.models_dir:
    config.substitutions.insert(0, ('%models_dir', config.models_dir))

config.substitutions.insert(0, ('%FileCheck', config.filecheck_path))

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.binary_dir, 'tests')
