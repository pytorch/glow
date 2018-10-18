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
#ifndef GLOW_ONNXIFI_THREAD_POOL_H
#define GLOW_ONNXIFI_THREAD_POOL_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace glow {
namespace onnxifi {

/// Thread pool for asynchronous execution of generic functions.
class ThreadPool final {
public:
  /// Constructor. Initializes a thread pool with \p numWorkers
  /// threads and has them all run ThreadPool::threadPoolWorkerMain.
  ThreadPool(unsigned numWorkers = kNumWorkers);

  /// Destructor. Signals to all threads to stop and waits for all of them
  /// to exit.
  ~ThreadPool();

  /// Submit \p fn as a work item for the thread pool.
  void submit(const std::function<void(void)> &fn);

private:
  /// Main loop run by the workers in the thread pool.
  void threadPoolWorkerMain();

  /// The default number of workers in the thread pool (overridable).
  constexpr static unsigned kNumWorkers = 10;

  /// Flag checked by the workers in between work items to determine
  /// whether they should stop and exit.
  std::atomic<bool> shouldStop_;

  /// Queue of work items.
  std::queue<std::function<void(void)>> workQueue_;

  /// Mutex to coordinate access to the work queue.
  std::mutex workQueueMtx_;

  /// Condition variable to signal to threads when work is added to
  /// the work queue.
  std::condition_variable queueNotEmpty_;

  /// Vector of worker thread objects.
  std::vector<std::thread> workers_;
};
} // namespace onnxifi
} // namespace glow

#endif // GLOW_ONNXIFI_THREAD_POOL_H
