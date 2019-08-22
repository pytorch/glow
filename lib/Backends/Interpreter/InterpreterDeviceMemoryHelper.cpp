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
#include "InterpreterDeviceMemoryHelper.h"

using namespace glow;

InterpreterDeviceMemoryHelper::InterpreterDeviceMemoryHelper() {}

InterpreterDeviceMemoryHelper::~InterpreterDeviceMemoryHelper() {
  for (const auto &pair : tensors_) {
    delete pair.second;
  }
  tensors_.clear();
}

void InterpreterDeviceMemoryHelper::allocateTensor(channel_id_type channelId,
                                                   TypeRef tensorType) {
  std::lock_guard<std::mutex> tGuard(tensorsGuard_);
  std::lock_guard<std::mutex> sGuard(transferStatusGuard_);
  assert(tensors_.find(channelId) == tensors_.end() &&
         "Tensor already allocated for channel ID");
  tensors_.insert(std::make_pair(channelId, new Tensor(tensorType)));
  transferStatus_.insert(std::make_pair(channelId, false));
}

Tensor *
InterpreterDeviceMemoryHelper::getLocalTensor(channel_id_type channelId) {
  std::lock_guard<std::mutex> tGuard(tensorsGuard_);
  auto it = tensors_.find(channelId);
  assert(it != tensors_.end() && "Unknown channel ID");
  return it->second;
}

Tensor *
InterpreterDeviceMemoryHelper::getRemoteTensor(channel_id_type channelId) {
  std::lock_guard<std::mutex> rGuard(remotesGuard_);
  auto it = remoteReceiverDevices_.find(channelId);
  assert(it != remoteReceiverDevices_.end() && "Unknown remote channel ID");
  return it->second->getLocalTensor(channelId);
}

bool InterpreterDeviceMemoryHelper::getLocalStatus(channel_id_type channelId) {
  const auto st = transferStatus_.find(channelId);
  assert(st != transferStatus_.end() && "Unknown channel ID");
  return st->second;
}

void InterpreterDeviceMemoryHelper::addRemoteReceiver(
    channel_id_type channelId, InterpreterDeviceMemoryHelper *receiver) {
  std::lock_guard<std::mutex> rGuard(remotesGuard_);
  remoteReceiverDevices_.insert(std::make_pair(channelId, receiver));
}

void InterpreterDeviceMemoryHelper::sendTensor(channel_id_type channelId,
                                               Tensor *message) {
  // copy data
  Tensor *dest = getRemoteTensor(channelId);
  dest->copyRawFrom(message);
  // set transfer status
  std::lock_guard<std::mutex> rGuard(remotesGuard_);
  auto it = remoteReceiverDevices_.find(channelId);
  assert(it != remoteReceiverDevices_.end() && "Unknown remote channel ID");
  it->second->transferCompleted(channelId);
}

void InterpreterDeviceMemoryHelper::transferCompleted(
    channel_id_type channelId) {
  std::lock_guard<std::mutex> sGuard(transferStatusGuard_);
  auto st = transferStatus_.find(channelId);
  assert(st != transferStatus_.end() && "Unknown channel ID");
  st->second = true;
}
