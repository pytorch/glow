# Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

set(VERIF_FILENAME NNPISpecificNodesVerification.h)
configure_file(${VERIF_FILENAME} ${GLOW_BINARY_DIR}/glow/${VERIF_FILENAME} COPYONLY)
configure_file(NNPISpecificTypes.h ${GLOW_BINARY_DIR}/glow/NNPISpecificTypes.h COPYONLY)

