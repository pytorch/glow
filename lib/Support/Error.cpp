/**
 * Copyright (c) 2017-present, Facebook, Inc.
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

#include "glow/Support/Error.h"

namespace glow {
llvm::ExitOnError exitOnErr("Encountered an error, exiting.\n");

/// ID used by llvm::ErrorInfo::isA's dynamic typing.
uint8_t const GlowErr::ID = 0;

bool OneErrOnly::set(llvm::Error err) {
  // Don't do anything in the case of empty Error.
  if (!err) {
    return false;
  }

  std::unique_lock<std::mutex> lock(m_);

  if (!err_) {
    err_ = std::move(err);
    return true;
  } else {
    // No update happening so don't need the lock any more.
    lock.unlock();
    LOG(ERROR) << "OneErrOnly already has an Error, discarding new Error: "
               << llvm::toString(std::move(err));
    return false;
  }
}

llvm::Error OneErrOnly::get() {
  std::unique_lock<std::mutex> lock(m_);
  auto err = std::move(err_);
  return err;
}

bool OneErrOnly::containsErr() {
  std::unique_lock<std::mutex> lock(m_);
  return static_cast<bool>(err_);
}
} // namespace glow
