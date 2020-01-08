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

#include "glow/Base/SharedMutex.h"
#include <cassert>

namespace glow {

shared_mutex::Event::Event(bool autoReset, bool signaled)
    : autoReset_(autoReset), signaled_(signaled) {}

void shared_mutex::Event::reset() {
  std::unique_lock<std::mutex> guard(mtx_);
  signaled_ = false;
}

void shared_mutex::Event::notify() {
  std::unique_lock<std::mutex> guard(mtx_);
  signaled_ = true;
  guard.unlock();
  cond_.notify_one();
}

void shared_mutex::Event::broadcast() {
  std::unique_lock<std::mutex> guard(mtx_);
  signaled_ = true;
  guard.unlock();
  cond_.notify_all();
}

void shared_mutex::Event::wait() {
  std::unique_lock<std::mutex> guard(mtx_);

  if (signaled_) { // Wait only if Event is in non-signaled state.
    return;
  }

  cond_.wait(guard, [this] { return signaled_; });

  if (autoReset_) { // Reset signaled state in manual reset mode.
    signaled_ = false;
  }
}

void shared_mutex::lock() {
  while (true) { // Loop until we get writer lock.
    bool locked = false;
    size_t readers = 0;
    std::unique_lock<std::mutex> guard(lock_);
    if (!released_) { // Already locked by another writer.
      // Remember the reason we can't acquire the write lock.
      locked = true;
    } else {
      released_ = false;   // Lock the door for readers.
      evOpen_.reset();     // Reset event and force new coming readers to wait.
      if (readers_ == 0) { // Lock is secure.
        break;             // Success.
      }
      // Remember the reason we can't acquire the write lock.
      readers = readers_;
    }
    guard.unlock();

    // Analyze the reason why write lock wasn't yet acquired.
    if (locked) { // Waiting for another writer.
      evCanLock_.wait();
      break;
    } else if (readers != 0) { // Some readers are still there.
      evCanClose_.wait();
      break;
    }
  }

  evCanClose_.reset(); // Block access for all coming readers.
}

bool shared_mutex::try_lock() {
  std::unique_lock<std::mutex> guard(lock_);
  if (released_ && readers_ == 0) { // Can lock.
    released_ = false;              // Lock the door.
    evOpen_.reset(); // Reset Event and force new coming readers to wait.
    return true;
  }
  return false;
}

void shared_mutex::unlock() {
  std::unique_lock<std::mutex> guard(lock_);
  if (released_) { // Already unlocked.
    return;
  }

  released_ = true;    // Unlock the door.
  evOpen_.broadcast(); // Notifies all readers who are waiting for access.
  evCanLock_.notify(); // Wake up only one writer.
}

void shared_mutex::lock_shared() {
  while (true) { // Loop until we get reader access.
    std::unique_lock<std::mutex> guard(lock_);
    if (released_) { // Can lock.
      ++readers_;
      return;
    }
    guard.unlock();

    evOpen_.wait(); // Wait for write lock release.
  }
}

bool shared_mutex::try_lock_shared() {
  std::unique_lock<std::mutex> guard(lock_);
  if (released_) { // Can lock.
    ++readers_;
    return true;
  }
  return false;
}

void shared_mutex::unlock_shared() {
  size_t readers = 0;
  bool closed = false;

  std::unique_lock<std::mutex> guard(lock_);
  assert(readers_ > 0);
  readers = --readers_;
  closed = !released_;

  if (closed && readers == 0) { // Last reader has left.
    evCanClose_.notify();       // Let first in the line writer go through.
  }
}

shared_mutex *shared_mutex::native_handle() { return this; }
} // namespace glow
