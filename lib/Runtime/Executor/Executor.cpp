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

#include "glow/Runtime/Executor/Executor.h"
#include "ThreadPoolExecutor.h"
#include "llvm/Support/Casting.h"

namespace glow {
namespace runtime {

Executor *createExecutor(const DeviceManagerMapTy &deviceManagers,
                         ExecutorKind executorKind) {
  switch (executorKind) {
  case ExecutorKind::ThreadPool:
    return new ThreadPoolExecutor(deviceManagers);
  }

  // This is to make compiler happy. It can never reach this point as the switch
  // statement above always covers all possible values.
  llvm_unreachable("unreachable");
}

} // namespace runtime
} // namespace glow
