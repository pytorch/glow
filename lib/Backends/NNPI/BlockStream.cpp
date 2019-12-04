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

#include "BlockStream.h"
#include "DebugMacros.h"
#include <cstring>

namespace glow {

BlockStream::BlockStream(size_t preallocSize, uint32_t pageBlockSize)
    : readOffest_(0), writeOffset_(0), blockSize_(pageBlockSize) {
  allocateBlocks(preallocSize);
}

BlockStream::~BlockStream() { releaseMemory(); }

size_t BlockStream::write(const char *buffer, size_t size) {
  // Make sure we have space to copy.
  if (allocateBlocks(size) < size) {
    return 0;
  }

  const char *bStart = &buffer[0];
  size_t copied = 0;
  size_t offsetInBlock = writeOffset_ % blockSize_;
  size_t blockIndex = writeOffset_ / blockSize_;
  while (size - copied > 0) {
    std::vector<char> &currentBlock = blocks_[blockIndex++];
    size_t blockCopySize =
        std::min(blockSize_ - offsetInBlock, (size - copied));
    const char *bEnd = &bStart[blockCopySize];
    auto dstBlockIt = std::back_inserter(currentBlock);
    std::copy(bStart, bEnd, dstBlockIt);
    bStart = bEnd;
    copied += blockCopySize;
    offsetInBlock = 0;
  }
  writeOffset_ += copied;
  return copied;
}

size_t BlockStream::read(char *buffer, size_t size) {
  char *bStart = &buffer[0];
  size_t readBytes = 0;
  size_t maxReadSize = writeOffset_ - readOffest_;
  size_t bytesToRead = std::min(maxReadSize, size);
  size_t offsetInBlock = readOffest_ % blockSize_;
  size_t blockIndex = readOffest_ / blockSize_;

  while (bytesToRead - readBytes > 0) {
    std::vector<char> &currentBlock = blocks_[blockIndex++];
    size_t blockCopySize =
        std::min(blockSize_ - offsetInBlock, (bytesToRead - readBytes));
    auto srcStartIt = currentBlock.begin() + offsetInBlock;
    auto srcEndIt = srcStartIt + blockCopySize;
    std::copy(srcStartIt, srcEndIt, bStart);
    bStart = &bStart[blockCopySize];
    readBytes += blockCopySize;
    offsetInBlock = 0;
  }
  readOffest_ += readBytes;
  return readBytes;
}

size_t BlockStream::getFreeAllocatedSpace() {
  return blockSize_ * blocks_.size() - writeOffset_;
}

size_t BlockStream::allocateBlocks(size_t size) {
  size_t available = getFreeAllocatedSpace();
  if (available > size) {
    // No need to allocate new blocks.
    return available;
  }
  int64_t missingSize = size - available;

  while (missingSize > 0) {
    std::vector<char> block;
    block.reserve(blockSize_);
    size_t reserved = block.capacity();
    if (reserved < blockSize_ && static_cast<int64_t>(reserved) < missingSize) {
      // Failed to allocate.
      return available;
    }
    available += reserved;
    missingSize -= reserved;
    blocks_.push_back(block);
  }
  return available;
}

void BlockStream::resetWrite() {
  for (auto &block : blocks_) {
    block.clear();
  }
  writeOffset_ = 0;
}

void BlockStream::releaseMemory() {
  blocks_.clear();
  blocks_.shrink_to_fit();
  reset();
}

} // namespace glow
