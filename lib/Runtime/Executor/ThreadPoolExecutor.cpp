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

#include "ThreadPoolExecutor.h"

namespace glow {
namespace runtime {

ThreadPoolExecutorWorkItem::ThreadPoolExecutorWorkItem(
    ExecutorFunctionDAG *dag, ExecutorFunctionDAGContext *ctx, DoneCb cb)
    : cb_(cb), dag_(dag), ctx_(ctx), status_(Status::NONE),
      result_(new Context()) {
  it_ = (dag_->functions).begin();
}

bool ThreadPoolExecutorWorkItem::isMoreWork() {
  std::lock_guard<std::mutex> lock(mtx_);
  // As long as the work item is in statued QUEUED, IN_PROGRESS, or FAILED,
  // there is more work to be done (i.e. the caller should bother calling
  // getNext()).
  return (status_ != Status::DONE && status_ != Status::NONE);
}

std::tuple<Function *, Context *> ThreadPoolExecutorWorkItem::getNext() {
  std::unique_lock<std::mutex> lock(mtx_);

  bool validState = status_ == Status::QUEUED || status_ == Status::IN_PROGRESS;

  // If the item is not in state QUEUED or IN_PROGRESS, there is no function/
  // context pair to return.
  if (!validState) {
    // If the item has been marked as FAILED, move it to DONE state and
    // invoke the callback.
    if (status_ == Status::FAILED) {
      status_ = Status::DONE;
      lock.unlock();
      cb_(ResultCode::FAILED, nullptr);
    }
    return std::make_tuple(nullptr, nullptr);
  }

  // Process any updates that were made since the last call to getNext() in
  // order to get updated information on work item state.
  processUpdates();

  // If all items are done, move the work item to DONE state and call the
  // callback with the result context.
  bool allDone = completedFunctions_.size() == (dag_->functions).size();
  if (allDone) {
    status_ = Status::DONE;
    lock.unlock();
    cb_(ResultCode::EXECUTED, result_);
    return std::make_tuple(nullptr, nullptr);
  }

  // If execution reaches this point, that means there are still unfinished
  // functions in this work item. However, they could be executing right now.
  // In any case, update the status of the work item to IN_PROGRESS.
  status_ = Status::IN_PROGRESS;
  auto currentIt = it_;

  // Scan through the list of functions and find one that is not executing
  // whose prerequisites are done.
  do {
    // If the iterator has reached the end of the list, reset it.
    if (it_ == (dag_->functions).end()) {
      it_ = (dag_->functions).begin();
    }

    Function *f = *it_;

    // Check if all prerequisites of the current candidate function are done.
    std::list<Function *> &prerequisites = (dag_->incoming).at(f);
    bool allPrerequisitesFinished = true;
    for (auto &prerequisite : prerequisites) {
      if (!completedFunctions_.count(prerequisite)) {
        allPrerequisitesFinished = false;
        break;
      }
    }

    // If all prerequisites are done and the function is not currently being
    // executed, record that it is now executing and return it.
    if (allPrerequisitesFinished && !inflightFunctions_.count(f)) {
      inflightFunctions_.insert(f);
      Context *ctx = (ctx_->contexts).at(f);
      return std::make_tuple(f, ctx);
    } else {
      ++it_;
    }

  } while (it_ != currentIt);

  // If we make one pass through the list of functions and find there is
  // nothing to run, return nothing.
  return std::make_tuple(nullptr, nullptr);
}

void ThreadPoolExecutorWorkItem::markSuccess(Function *function,
                                             Context *context) {
  std::lock_guard<std::mutex> lock(mtx_);
  updateFunctions_.insert(function);
  updateContexts_.insert(context);
}

void ThreadPoolExecutorWorkItem::markQueued() {
  std::lock_guard<std::mutex> lock(mtx_);
  status_ = Status::QUEUED;
}

void ThreadPoolExecutorWorkItem::markFailure() {
  std::lock_guard<std::mutex> lock(mtx_);
  status_ = Status::FAILED;
}

void ThreadPoolExecutorWorkItem::processUpdates() {
  auto fnIt = updateFunctions_.begin();
  auto fnEnd = updateFunctions_.end();
  auto ctxIt = updateContexts_.begin();
  auto ctxEnd = updateContexts_.end();

  while ((fnIt != fnEnd) && (ctxIt != ctxEnd)) {
    Function *f = *fnIt;
    Context *ctx = *ctxIt;

    // For every completed function, copy its outputs to the Context of any
    // of the functions that depend on it that need that output.
    std::list<std::string> &outputs = (dag_->outputs).at(f);
    std::list<Function *> &postrequisites = (dag_->outgoing).at(f);
    for (auto &output : outputs) {
      for (auto &postrequisite : postrequisites) {
        Module *postModule = postrequisite->getParent();
        Context *postCtx = (ctx_->contexts).at(postrequisite);
        Placeholder *p;
        if ((p = postModule->getPlaceholderByName(output))) {
          postCtx->insert(p, ctx->get(p)->clone());
        }
      }
    }

    // Mark the function as completed instead of inflight/executing.
    completedFunctions_.insert(f);
    inflightFunctions_.erase(f);
    ++ctxIt;
    ++fnIt;
  }
}

ThreadPoolExecutor::ThreadPoolExecutor(unsigned numWorkers) {
  // Intialize all workers and make each one run workerMain.
  for (unsigned i = 0; i < numWorkers; i++) {
    std::thread th(std::bind(&ThreadPoolExecutor::workerMain, this));
    workers_.emplace_back(std::move(th));
  }
}

void ThreadPoolExecutor::run(ExecutorFunctionDAG *functionDag,
                             ExecutorFunctionDAGContext *ctx, DoneCb cb) {
  // Create a new work item from the provided information.
  auto workItem = new ThreadPoolExecutorWorkItem(functionDag, ctx, cb);
  // Put the work item onto the queue and mark it as queued. Signal to any
  // worker waiting for work items.
  std::unique_lock<std::mutex> lock(workQueueMtx_);
  workQueue_.push(workItem);
  workItem->markQueued();
  lock.unlock();
  queueNotEmpty_.notify_one();
}

ThreadPoolExecutor::~ThreadPoolExecutor() {
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

void ThreadPoolExecutor::workerMain() {
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
    processWorkItem(workItem);
  }
}

void ThreadPoolExecutor::processWorkItem(ThreadPoolExecutorWorkItem *workItem) {
  // Check if there is more work left in this work item. If not, that means
  // it either succeeded or failed, and the callback
  if (workItem->isMoreWork()) {
    Function *f;
    Context *ctx;
    std::tie(f, ctx) = workItem->getNext();

    // If there is a function and context available to work on, run it.
    if (f && ctx) {
      someDeviceManagerFunction(
          f, ctx, [workItem, f](ResultCode resultCode, Context *ctx) {
            if (resultCode == ResultCode::EXECUTED) {
              workItem->markSuccess(f, ctx);
            } else if (resultCode == ResultCode::FAILED) {
              workItem->markFailure();
            }
          });
    }

    // If isMoreWork() returned true but getNext() returned nothing, this work
    // item has more work that needs doing but not until some dependencies are
    // fulfilled. Requeue it so that another worker will look at it again.
    std::unique_lock<std::mutex> lock(workQueueMtx_);
    workQueue_.push(workItem);
    lock.unlock();
    queueNotEmpty_.notify_one();
  }
}
} // namespace runtime
} // namespace glow
