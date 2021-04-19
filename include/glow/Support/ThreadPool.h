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
#ifndef GLOW_SUPPORT_THREADPOOL_H
#define GLOW_SUPPORT_THREADPOOL_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <set>
#include <thread>
#include <vector>

namespace glow {

namespace threads {
/// Returns a unique id associated with the current thread.
size_t getThreadId();

/// Returns a unique id associated with a new virtual thread (i.e. a device
/// tid).
size_t createThreadId();
} // namespace threads

#ifdef WIN32
/// A copyable wrapper for a lambda function that has non-copyable objects in
/// its lambda capture.
/// This is useful for VS builds where std::packaged_tasks wraps a
/// std::function which must be copyable.
template <class F> struct shared_function {
  std::shared_ptr<F> f;
  shared_function() = delete;
  shared_function(F &&f_) : f(std::make_shared<F>(std::move(f_))) {}
  shared_function(shared_function const &) = default;
  shared_function(shared_function &&) = default;
  shared_function &operator=(shared_function const &) = default;
  shared_function &operator=(shared_function &&) = default;
  template <class... As> auto operator()(As &&...as) const {
    return (*f)(std::forward<As>(as)...);
  }
};
template <class F>
shared_function<std::decay_t<F>> make_shared_function(F &&f) {
  return {std::forward<F>(f)};
}
#endif

/// An executor that runs Tasks on a single thread.
class ThreadExecutor final {
public:
  /// Constructor. Initializes one thread backed by the workQueue_.
  explicit ThreadExecutor(const std::string &name = "");

  /// Destructor. Signals the thread to stop and waits for exit.
  ~ThreadExecutor();

  /// Submit \p fn as a work item for the thread pool.
  /// \p fn must be a lambda with void return type and arguments.
  template <typename F> std::future<void> submit(F &&fn) {
    // Add fn to the work queue.
    std::unique_lock<std::mutex> lock(workQueueMtx_);

#ifdef WIN32
    std::packaged_task<void(void)> task(make_shared_function(std::move(fn)));
#else
    std::packaged_task<void(void)> task(std::move(fn));
#endif

    auto future = task.get_future();
    workQueue_.push(std::move(task));
    lock.unlock();
    queueNotEmpty_.notify_one();
    return future;
  }

  /// Submit \p task as a work item for the thread pool.
  std::future<void> submit(std::packaged_task<void(void)> &&task);

  void stop(bool block = false);

protected:
  /// Main loop run by the workers in the thread pool.
  void threadPoolWorkerMain();

  /// Flag checked in between work items to determine whether we should stop and
  /// exit.
  std::atomic<bool> shouldStop_{false};

  /// Queue of work items.
  std::queue<std::packaged_task<void(void)>> workQueue_;

  /// Mutex to coordinate access to the work queue.
  std::mutex workQueueMtx_;

  /// Condition variable to signal to threads when work is added to
  /// the work queue.
  std::condition_variable queueNotEmpty_;

  /// Worker thread.
  std::thread worker_;
};

/// Thread pool for asynchronous execution of generic functions.
class ThreadPool final {
public:
  /// Constructor. Initializes a thread pool with \p numWorkers
  /// threads and has them all run ThreadPool::threadPoolWorkerMain.
  ThreadPool(unsigned numWorkers = kNumWorkers, const std::string &name = "");

  /// Destructor. Signals to all threads to stop and waits for all of them
  /// to exit.
  ~ThreadPool();

  /// Stop all threads and optionally wait for them to join.
  void stop(bool block = false);

  /// Submit \p fn as a work item for the thread pool.
  /// \p fn must be a lambda with void return type and arguments.
  template <typename F> std::future<void> submit(F &&fn) {
#ifdef WIN32
    std::packaged_task<void(void)> task(make_shared_function(std::move(fn)));
#else
    std::packaged_task<void(void)> task(std::move(fn));
#endif

    return submit(std::move(task));
  }

  /// Submit \p task as a work item for the thread pool.
  std::future<void> submit(std::packaged_task<void(void)> &&task);

  /// Returns a ThreadExecutor that can be accessed directly, allowing
  /// submitting multiple tasks to the same thread.
  ThreadExecutor *getExecutor() {
    size_t exIndex = nextWorker_++;
    return workers_[exIndex % workers_.size()];
  }

  /// Run the provided function on every thread in the ThreadPool. The function
  /// must be copyable.
  template <typename F> std::future<void> runOnAllThreads(F &&fn) {
    std::shared_ptr<std::atomic<size_t>> finished =
        std::make_shared<std::atomic<size_t>>(0);
    std::shared_ptr<std::promise<void>> promise =
        std::make_shared<std::promise<void>>();
    for (auto *w : workers_) {
      w->submit([fn, finished, promise, total = workers_.size()]() {
        fn();
        if ((finished->fetch_add(1) + 1) >= total) {
          promise->set_value();
        }
      });
    }

    return promise->get_future();
  }

  const std::set<size_t> &getThreadIds() { return threadIds_; }

private:
  /// The default number of workers in the thread pool (overridable).
  constexpr static unsigned kNumWorkers = 10;

  /// Vector of worker thread objects.
  /// It is safe to access this without a lock as it is const after
  /// construction.
  std::vector<ThreadExecutor *> workers_;

  /// Round robin index for the next work thread.
  std::atomic<size_t> nextWorker_{0};

  /// Thread Ids and associated names owned by this ThreadPool.
  std::set<size_t> threadIds_;
};
} // namespace glow

#endif // GLOW_SUPPORT_THREADPOOL_H
