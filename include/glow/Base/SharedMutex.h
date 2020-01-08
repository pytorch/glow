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
#ifndef GLOW_BASE_SHAREDMUTEX_H
#define GLOW_BASE_SHAREDMUTEX_H

#include <condition_variable>
#include <mutex>

namespace glow {

/// Glow implementation of std shared_mutex for C++ 14.
/// Replace glow::shared_mutex with std::shared_mutex in C++ 17.
/// Class name violates a name convension but make it a drop-in replacement
/// for c++17 std::shared_mutex.
class shared_mutex {
  /// Private class Event based on std::mutex and std::conditional_variable.
  class Event {
  public:
    /// Constructor sets initial state (signaled, non-signaled) and
    /// auto/manual reset.
    explicit Event(bool autoReset = true, bool signaled = false);

    /// Set Event object into signaled state. Only one wait() will be waken up.
    /// If Event object is already in signal state, method does nothing.
    void notify();

    /// Set Event object into signaled state. All wait() will be waken up.
    /// If Event object is already in signal state, method does nothing.
    void broadcast();

    /// Set Event object into non-signaled state explicitly (manual).
    /// If Event object is already in non-signal state, method does nothing.
    void reset();

    /// Unconditional wait for Event to be set into signal state.
    void wait();

  private:
    /// Internal obects.
    std::mutex mtx_;               /// STL mutex.
    std::condition_variable cond_; /// STL conditional variable.
    bool autoReset_;               /// Reset flag.
    bool signaled_;                /// Signal flag.
  };

  Event evOpen_{false, true};      /// Manual reset, signaled state.
  Event evCanClose_{false, false}; /// Manual reset, non signaled state.
  Event evCanLock_{true, false};   /// Auto reset, non signaled state.

  bool released_{true}; /// Is resource released by all write-locks.
  size_t readers_{0};   /// Number of active read-locks.

  std::mutex lock_; /// Protects access to the internal resources.
public:
  shared_mutex() = default;
  shared_mutex(const shared_mutex &) = delete;
  shared_mutex &operator=(const shared_mutex &) = delete;

  void lock();
  bool try_lock();
  void unlock();

  void lock_shared();
  bool try_lock_shared();
  void unlock_shared();

  shared_mutex *native_handle();
};

/// Useful RTTI classes for shared_mutex
/// Class name violates a name convension but make it a drop-in replacement
/// for c++17 std::shared_lock.
template <typename SM> class shared_lock {
  SM &mtx_;
  bool locked_{false};
  size_t readers_{0};

public:
  /// No initial lock constructor.
  explicit shared_lock(SM &mtx) : mtx_(mtx) {}

  explicit shared_lock(SM &mtx, bool writeLock) : mtx_(mtx) {
    writeLock ? try_lock() : try_lock_shared();
  }

  ~shared_lock() {
    if (locked_) {
      unlock();
    } else {
      while (readers_ > 0) {
        unlock_shared();
      }
    }
  }

  /// Write locks.
  void lock() {
    mtx_.lock();
    locked_ = true;
  }

  bool try_lock() {
    if (!mtx_.try_lock()) {
      return false;
    }
    locked_ = true;
    return true;
  }

  void unlock() {
    mtx_.unlock();
    locked_ = false;
  }

  /// Read locks.
  void lock_shared() {
    mtx_.lock_shared();
    ++readers_;
  }

  bool try_lock_shared() {
    if (!mtx_.try_lock_shared()) {
      return false;
    }
    ++readers_;
    return true;
  }

  void unlock_shared() {
    mtx_.unlock_shared();
    --readers_;
  }
};

} // namespace glow

#endif // GLOW_BASE_SHAREDMUTEX_H
