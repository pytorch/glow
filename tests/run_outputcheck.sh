#!/bin/bash

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

set -uxo pipefail

# Make a temporary file to store the program output in
TEMP_FILE="$(mktemp)"

# Save the output of the program to TEMP_FILE
$1 > $TEMP_FILE
RUN_RESULT=$?

# If the program failed then print the output and quit
if [ $RUN_RESULT != 0 ]
then
  cat $TEMP_FILE
  rm $TEMP_FILE
  exit $RUN_RESULT
fi

# Run OutputCheck on the output of the program
cat $TEMP_FILE | $OUTPUTCHECK $1
CHECK_RESULT=$?

# If the check failed then print the output and quit
if [ $CHECK_RESULT != 0 ]
then
  cat $TEMP_FILE
  rm $TEMP_FILE
  exit $CHECK_RESULT
fi