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
#include "glow/Support/ThreadPool.h"

namespace glow {

ThreadExecutor::ThreadExecutor()
    : shouldStop_(false), worker_([this]() { threadPoolWorkerMain(); }) {}

ThreadExecutor::~ThreadExecutor() { stop(true); }

void ThreadExecutor::stop(bool block) {
  // Lock mutex before signalling for threads to stop to make sure
  //   // a thread can't wait on the condition variable after checking the
  //     // *old* value of shouldStop_.
  std::unique_lock<std::mutex> lock(workQueueMtx_);

  shouldStop_ = true;
  lock.unlock();
  queueNotEmpty_.notify_all();

  if (block && worker_.joinable()) {
    worker_.join();
  }
}

void ThreadExecutor::threadPoolWorkerMain() {
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
    auto workItem = std::move(workQueue_.front());
    workQueue_.pop();
    lock.unlock();

    // Process work item.
    workItem();
  }
}

std::future<void>
ThreadExecutor::submit(std::packaged_task<void(void)> &&task) {
  std::unique_lock<std::mutex> lock(workQueueMtx_);
  auto future = task.get_future();
  workQueue_.push(std::move(task));
  lock.unlock();
  queueNotEmpty_.notify_one();
  return future;
}

ThreadPool::ThreadPool(unsigned numWorkers) {
  // Intialize all workers and make each one run threadPoolWorkerMain.
  workers_.reserve(kNumWorkers);
  for (unsigned i = 0; i < numWorkers; i++) {
    workers_.push_back(new ThreadExecutor());
  }
}

ThreadPool::~ThreadPool() {
  stop(true);
  for (auto *w : workers_) {
    delete w;
  }
  workers_.clear();
}

void ThreadPool::stop(bool block) {
  // Signal to workers to stop.
  for (auto *w : workers_) {
    w->stop(block);
  }
}

std::future<void> ThreadPool::submit(std::packaged_task<void(void)> &&task) {
  ThreadExecutor *ex = getExecutor();
  return ex->submit(std::move(task));
}

} // namespace glow
