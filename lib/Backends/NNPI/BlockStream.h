/*
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

#ifndef GLOW_NNPI_BLOCK_STREAM_H
#define GLOW_NNPI_BLOCK_STREAM_H
#include "glow/Backend/BlockStreamBase.h"
#include <memory>
#include <vector>
namespace glow {

#define DEFAULT_BLOCK_STREAM_BLOCK_SIZE (1024 * 1024 * 10)
class BlockStream : public BlockStreamBase {
public:
  /// Default constructor with 0 pre-allocated blocks and 10M page size.
  BlockStream() : BlockStream(0, DEFAULT_BLOCK_STREAM_BLOCK_SIZE) {}
  /// Constructor with preallocSize pre-allocated memory and 10M page size.
  explicit BlockStream(size_t preallocSize)
      : BlockStream(preallocSize, DEFAULT_BLOCK_STREAM_BLOCK_SIZE) {}
  /// Constructor with preallocSize pre-allocated memory and pageBlockSize page
  /// size.
  BlockStream(size_t preallocSize, uint32_t pageBlockSize);

  /// Destructor: cleans up the allocated blocks.
  virtual ~BlockStream();

  /// Read from stream (updates read offset).
  size_t read(char *buffer, size_t size);
  /// Read to stream (updates read offset).
  size_t write(const char *buffer, size_t size);
  /// Get size of written data.
  size_t getSize() { return writeOffset_; }
  /// Reset read offset (not changing steam data - can re-read).
  void resetRead() { readOffest_ = 0; }
  /// Reset write offset (not deleting allocated memory).
  void resetWrite();
  /// Reset both read and write offset (not deleting allocated memory).
  void reset() {
    resetRead();
    resetWrite();
  }
  /// Release allocated memory and reset.
  void releaseMemory();

private:
  uint64_t readOffest_;
  uint64_t writeOffset_;
  size_t blockSize_;
  std::vector<std::vector<char>> blocks_;

  size_t getFreeAllocatedSpace();
  size_t allocateBlocks(size_t size);
};

} // namespace glow

#endif // GLOW_NNPI_BLOCK_STREAM_H
