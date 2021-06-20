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
#ifndef GLOW_BACKENDS_BLOCKSTREAMBASE_H
#define GLOW_BACKENDS_BLOCKSTREAMBASE_H

#include <vector>
namespace glow {

// This is the base class of all BlockStream implemented in backends.
// BlockStream are used for read & write compiled backend function.
class BlockStreamBase {
public:
  BlockStreamBase() {}
  virtual ~BlockStreamBase() {}

  // Read /p size bytes from stream to /p buffer.
  virtual size_t read(char *buffer, size_t size) = 0;

  // Write /p size bytes from /p buffer to stream.
  virtual size_t write(const char *buffer, size_t size) = 0;

  // Get size of wriiten data.
  virtual size_t getSize() = 0;

  // Release memory of block stream
  virtual void releaseMemory() = 0;
};

} // end namespace glow

#endif // GLOW_BACKENDS_BLOCKSTREAMBASE_H
