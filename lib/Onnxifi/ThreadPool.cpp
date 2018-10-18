/*
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
#include "ThreadPool.h"

namespace glow {
namespace onnxifi {

ThreadPool::ThreadPool(unsigned numWorkers) : shouldStop_(false) {
  // Intialize all workers and make each one run threadPoolWorkerMain.
  for (unsigned i = 0; i < numWorkers; i++) {
    std::thread th(std::bind(&ThreadPool::threadPoolWorkerMain, this));
    workers_.push_back(std::move(th));
  }
}

ThreadPool::~ThreadPool() {
  // Lock mutex before signalling for threads to stop to make sure
  // a thread can't wait on the condition variable after checking the
  // *old* value of shouldStop_.
  std::unique_lock<std::mutex> lock(workQueueMtx_);

  // Signal to workers to stop.
  shouldStop_ = true;

  // Notify all worker threads in case any are waiting on the condition
  // variable.
  lock.unlock();
  queueNotEmpty_.notify_all();

  // Join all worker threads.
  for (auto &w : workers_) {
    w.join();
  }
  workers_.clear();
}

void ThreadPool::submit(const std::function<void(void)> &fn) {
  // Add fn to the work queue.
  std::unique_lock<std::mutex> lock(workQueueMtx_);
  workQueue_.push(fn);
  lock.unlock();
  queueNotEmpty_.notify_one();
}

void ThreadPool::threadPoolWorkerMain() {
  std::unique_lock<std::mutex> lock(workQueueMtx_, std::defer_lock);

  while (!shouldStop_) {
    // Lock the lock after processing a work item.
    lock.lock();

    // If work queue is empty, wait to be signalled when
    // a work item is submitted.
    while (workQueue_.empty() && !shouldStop_) {
      queueNotEmpty_.wait(lock);
    }

    // If shouldStop_ was set to false while the thread
    // was asleep, break out of the main loop.
    if (shouldStop_) {
      break;
    }

    // Pop a work item from the queue, and make sure to unlock
    // the lock before processing it.
    auto workItem = workQueue_.front();
    workQueue_.pop();
    lock.unlock();

    // Process work item.
    workItem();
  }
}
} // namespace onnxifi
} // namespace glow
