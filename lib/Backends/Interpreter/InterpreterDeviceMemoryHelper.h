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
#ifndef GLOW_BACKENDS_INTERPRETER_INTERPRETERDEVICEMEMORYHELPER_H
#define GLOW_BACKENDS_INTERPRETER_INTERPRETERDEVICEMEMORYHELPER_H

#include "glow/Base/Tensor.h"
#include <mutex>
#include <unordered_map>

namespace glow {

typedef int64_t channel_id_type;

class InterpreterDeviceMemoryHelper final {

public:
  explicit InterpreterDeviceMemoryHelper();
  ~InterpreterDeviceMemoryHelper();

  /// Allocates tensor memory on local device
  void allocateTensor(channel_id_type channelId, TypeRef tensorType);
  /// Add remote receiver device mapped to the given channelId
  void addRemoteReceiver(channel_id_type channelId,
                         InterpreterDeviceMemoryHelper *receiver);
  /// Sends tensor data to the remote receiver device corresponding to channelId
  void sendTensor(channel_id_type channelId, Tensor *message);
  /// Returns local tensor backing
  Tensor *getLocalTensor(channel_id_type channelId);
  /// Returns true if the tensor was received
  bool getLocalStatus(channel_id_type channelId);

private:
  Tensor *getRemoteTensor(channel_id_type channelId);
  void transferCompleted(channel_id_type channelId);

  std::unordered_map<channel_id_type, Tensor *> tensors_; // received tensors
  std::mutex tensorsGuard_;
  std::unordered_map<channel_id_type, InterpreterDeviceMemoryHelper *>
      remoteReceiverDevices_; // devices to send tensors to
  std::mutex remotesGuard_;
  std::unordered_map<channel_id_type, bool> transferStatus_;
  std::mutex transferStatusGuard_;
};

} // namespace glow
#endif // GLOW_BACKENDS_INTERPRETER_INTERPRETERDEVICEMEMORYHELPER_H
