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

#ifndef _GLOW_BACKENDS_INTERPRETER_INTERPRETER_CUSTOM_OP_H
#define _GLOW_BACKENDS_INTERPRETER_INTERPRETER_CUSTOM_OP_H

#include "glow/Backend/BackendOpRepository.h"
#include "glow/CustomOp/CustomOpInterpreterInterface.h"
#include "glow/Graph/CustomOpUtils.h"
#include "glow/Graph/OperationInfo.h"

#include <unordered_map>

namespace glow {

class InterpreterImplInfo : public ImplementationInfo {
private:
  std::string path_{};
  void *handle_{};
  bool isLoaded_{false};

  // Execution kernel pointer.
  customOpInterpreterKernel_t kernel_{};

public:
  InterpreterImplInfo(const ImplementationInfo &opInfo)
      : ImplementationInfo(opInfo),
        path_(*(std::string *)opInfo.getImplementation()) {}

  // Loads the shared library if not already loaded.
  // Looks for the execution kernel symbol based on flavour ID.
  Error loadImplLibrary();
  void releaseImplLibrary();
  ~InterpreterImplInfo() {}

  customOpInterpreterKernel_t getKernel() {
    if (!isLoaded_)
      return nullptr;
    return kernel_;
  }
};

class InterpreterOpRepository : public BackendOpRepository {
private:
  // Map of all implementations for all ops.
  // Key is formed as "<Package>::<OpType>::<FlavourID>".
  std::unordered_map<std::string, InterpreterImplInfo> implementationMap_;

  // Map of selection functions.
  // Key is formed as "<Package>::<OpType>".
  std::unordered_map<std::string, customOpSelectImpl_t> selectionFuncsMap_;

public:
  std::string getBackendOpRepositoryName() const override {
    return "Interpreter";
  }
  static std::string getName() { return "Interpreter"; }

  // Adds the selection function and each implementation for this operator to
  // corresponding maps. Each ImplementationInfo is wrapped by
  // InterpreterImplInfo and loadImplLibrary is called on it.
  Error registerImplementations(llvm::StringRef opTypeName,
                                llvm::StringRef opPackageName,
                                llvm::ArrayRef<ImplementationInfo> impls,
                                customOpSelectImpl_t implSelect) override;

  // Looks up the selectionFuncsMap_ and returns selection function pointer.
  Expected<customOpSelectImpl_t>
  getSelectionFunction(llvm::StringRef opTypeName,
                       llvm::StringRef opPackageName);
  // Looks up the implementationMap_ and returns implementation kernel.
  Expected<customOpInterpreterKernel_t>
  getImplementation(llvm::StringRef opTypeName, llvm::StringRef opPackageName,
                    llvm::StringRef flavourId);

  // Clear implementations.
  void clear() override;
};
} // end namespace glow

#endif
