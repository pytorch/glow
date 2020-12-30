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

#ifndef GLOW_GRAPH_OPREPOSITORY_H
#define GLOW_GRAPH_OPREPOSITORY_H

#include <map>

#include "glow/Base/Traits.h"
#include "glow/Base/Type.h"
#include "glow/CustomOp/CustomOpFunctions.h"
#include "glow/Graph/OperationInfo.h"
#include "glow/Support/Error.h"

namespace glow {

class ProcessedOperationInfo;

/// Singleton to Register Custom Operations.
/// TODO: Make thread-safe.
class OpRepository {
public:
  // Key pair of op typeName and packageName.
  using OpKey = std::pair<std::string, std::string>;

  /// Return a pointer to the instance of OpRepository type.
  static OpRepository *get() {
    if (sInstance_ == nullptr) {
      sInstance_.reset(new OpRepository());
    }
    return sInstance_.get();
  }

  /// Register a new custom \p opInfo .
  /// Returns an Error, if
  ///  1. OperationInfo with the same typeName and packageName has already
  ///  been registered
  ///  2. Implementation is provided for an unsupported backend or if the
  ///  backend is supported but it has not registered its BackendOpRepository
  ///  3. Fails to load the verification library.
  Error registerOperation(const OperationInfo &opInfo);

  /// Parses \p fileName to read OperationInfo.
  /// Then registers these operations.
  Error registerOperation(llvm::StringRef fileName);

  /// Returns true if OperationInfo with \p opTypeName and \p opPackageName
  /// has been registered.
  bool isOpRegistered(llvm::StringRef opTypeName,
                      llvm::StringRef opPackageName) const;

  /// Return a shared_ptr to ProcessedOperationInfo with \p opTypeName and \p
  /// opPackageName. Returns a nullptr if such an OperationInfo has not been
  /// registered.
  std::shared_ptr<ProcessedOperationInfo>
  getOperationInfo(llvm::StringRef opTypeName,
                   llvm::StringRef opPackageName) const;

  /// Get a map of registered operations
  const std::map<OpKey, std::shared_ptr<ProcessedOperationInfo>> &
  getOperationsMap() {
    return operationRepo_;
  }

  // Removes all the opInfos that were registered.
  // Removes all the implementations registered with the BackendOpRepositories.
  // Provides an option to clear the OpRepo and BackendOpRepo as needed by
  // the application. OpRepository is a singleton class and the destructor is
  // not invoked until the process ends.
  void clear();

private:
  static std::unique_ptr<OpRepository> sInstance_;

  OpRepository();

  OpKey makeOpKey(llvm::StringRef opTypeName,
                  llvm::StringRef opPackageName) const;
  std::map<OpKey, std::shared_ptr<ProcessedOperationInfo>> operationRepo_;

}; // class OpRepository

// Processed OperationInfo stores OperationInfo and processes it.
// 1. Registers Implementation information with BackendRepository.
// 2. Loads the library with functions required for custom op execution.
class ProcessedOperationInfo : public OperationInfo {
private:
  // Stores library info and function ptrs.
  struct FunctionInfo {
    // lib info.
    std::string path_{};
    void *handle_{};
    bool isLoaded_{false};

    // function ptrs.
    customOpVerify_t verify_{};
    customOpSelectImpl_t selectImpl_{};
    customOpInferShape_t inferShape_{};
  } functionInfo_;

  // Releases the shared library if it was successfuly loaded
  // earlier, by calling dlclose.
  // Returns 0 if successful.
  int releaseFunctionLibrary() const;

public:
  ProcessedOperationInfo(const OperationInfo &opInfo) : OperationInfo(opInfo) {}

  // Loads the shared library with functions if not already loaded.
  // Looks for the symbol for verification function and stores it.
  Error loadFunctionLibrary();

  // Registers implementations with the backend op repository.
  Error registerImplementations();

  // Returns the verification function ptr.
  // Tries to load the library if not already loaded.
  // Returns nullptr in case an error is encountered while loading the library.
  customOpVerify_t getVerificationFunction();

  // Returns ptr to shape inference function.
  // Tries to load the library if not already loaded.
  // Returns nullptr in case an error is encountered while loading the library.
  customOpInferShape_t getShapeInferenceFunction();

  ~ProcessedOperationInfo();

}; // class ProcessedOperationInfo

} // namespace glow

#endif // GLOW_GRAPH_OPREPOSITORY_H
