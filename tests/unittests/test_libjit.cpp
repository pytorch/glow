/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <map>

/// The following variable needs to be defined in libjit, because codegen looks
/// for it to determine the size of size_t on the target.
size_t libjit_sizeTVar;

/// Simply write a global map to validate that static C++ constructors are
/// correctly called.
extern std::map<float, float> GlobalMap;
std::map<float, float> GlobalMap;
void libjit_WriteGlobalMap() { GlobalMap[1.0] = 1.2f; }

extern "C" {

#define JIT_MAGIC_VALUE 555

// JIT test dispatch function.
__attribute__((noinline)) void libjit_JITTestDispatch(float *src, float *dest) {
  if (*src == 0.f) {
    libjit_WriteGlobalMap();
  } else {
    // No test for this index. This should be an error.
    *dest = 0;
  }

  *dest = JIT_MAGIC_VALUE + *src;
}
}
