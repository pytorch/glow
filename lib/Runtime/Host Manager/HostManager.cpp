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

#include "glow/Runtime/Host Manager/HostManager.h"

using namespace glow;

int HostManager::addNetwork(Function *F) {
  int networkID = 5;
  // partitions
  // provision
  return networkID;
}

void HostManager::removeNetwork(int networkID) {
  // remove from networks
  // walk DAG and remove from deviceManagers
}

bool HostManager::networkAdded(int networkID) {
  return networks_.find(networkID) != networks_.end();
}

void HostManager::clearHost() {}

bool HostManager::runNetwork(int networkID, llvm::StringRef functionName,
                             Context context) {
  // runme
}