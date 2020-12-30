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
#include <dlfcn.h>

#include "glow/Backend/Backend.h"
#include "glow/Backend/BackendOpRepository.h"
#include "glow/Graph/OpRepository.h"

namespace glow {

std::unique_ptr<OpRepository> OpRepository::sInstance_ = nullptr;

// Default constructor
OpRepository::OpRepository() {}

OpRepository::OpKey
OpRepository::makeOpKey(llvm::StringRef opTypeName,
                        llvm::StringRef opPackageName) const {
  return std::make_pair(opTypeName, opPackageName);
}

/// Register a new custom \p opInfo.
Error OpRepository::registerOperation(const OperationInfo &opInfo) {
  std::string type, package;
  type = opInfo.getTypeName();
  package = opInfo.getPackageName();
  OpKey opkey = makeOpKey(type, package);
  auto itr = operationRepo_.find(opkey);

  RETURN_ERR_IF_NOT(itr == operationRepo_.end(),
                    "Op already registered with type " + type +
                        " and package name " + package);

  // Create ProcessedOperationInfo and process the operation info
  // provided by the customer.
  std::shared_ptr<ProcessedOperationInfo> pOpInfo =
      std::make_shared<ProcessedOperationInfo>(opInfo);
  RETURN_IF_ERR(pOpInfo->loadFunctionLibrary());
  RETURN_IF_ERR(pOpInfo->registerImplementations());

  // If processing was successful store the ProcessedOperationInfo.
  operationRepo_.emplace(std::make_pair(opkey, pOpInfo));

  LOG(INFO) << "OpRepository: Registered OpInfo: Name "
            << pOpInfo->getTypeName().str() << " Package "
            << pOpInfo->getPackageName().str() << std::endl;
  return Error::success();
}

/// Returns true if OperationInfo with \p opTypeName and \p opPackageName
/// has been registered.
bool OpRepository::isOpRegistered(llvm::StringRef opTypeName,
                                  llvm::StringRef opPackageName) const {

  OpKey opkey = makeOpKey(opTypeName, opPackageName);
  auto itr = operationRepo_.find(opkey);
  return itr != operationRepo_.end();
}

/// Return a ptr to ProcessedOperationInfo with \p opTypeName and \p
/// opPackageName. Returns a nullptr if such an OperationInfo has not been
/// registered.
std::shared_ptr<ProcessedOperationInfo>
OpRepository::getOperationInfo(llvm::StringRef opTypeName,
                               llvm::StringRef opPackageName) const {
  OpKey opkey = makeOpKey(opTypeName, opPackageName);
  auto itr = operationRepo_.find(opkey);

  if (itr == operationRepo_.end()) {
    return nullptr;
  }
  return itr->second;
}

Error OpRepository::registerOperation(llvm::StringRef fileName) {
  std::vector<OperationInfo> opinfos;
  RETURN_IF_ERR(deserializeOpInfoFromYaml(fileName, opinfos));

  for (const auto &op : opinfos) {
    RETURN_IF_ERR(registerOperation(op));
  }

  return Error::success();
}

static void clearBackendOpRepository(llvm::StringRef backendName) {
  auto repo = getBackendOpRepository(backendName);
  repo->clear();
  return;
}

void OpRepository::clear() {
  std::vector<std::string> backendOpRepos = getAvailableBackendOpRepositories();
  for (const auto &backend : backendOpRepos)
    clearBackendOpRepository(backend);
  operationRepo_.clear();
  return;
}

//===----------------------------------------------------------------------===//
//                     ProcessedOperationInfo
//===----------------------------------------------------------------------===//

/// ~Dtor.
ProcessedOperationInfo::~ProcessedOperationInfo() {
  LOG(INFO) << "Destroying OpInfo: Name " << getTypeName().str() << " Package "
            << getPackageName().str() << std::endl;
  releaseFunctionLibrary();
}

/// Releases the shared library if it was successfuly loaded
/// earlier, by calling dlclose Returns 0 if successful.
int ProcessedOperationInfo::releaseFunctionLibrary() const {
  if (!functionInfo_.isLoaded_) {
    return 0;
  }

  int status = dlclose(functionInfo_.handle_);
  const char *dlerr = dlerror();
  if (status != 0) {
    LOG(ERROR) << "Error in dlclose the library " << functionInfo_.path_
               << ". Got error " << dlerr << std::endl;
  }
  return status;
}

// Loads the shared library if not already loaded.
// Looks for the symbol for verification function and stores it.
// Looks for the symbol for selection function and stores it.
Error ProcessedOperationInfo::loadFunctionLibrary() {
  if (functionInfo_.isLoaded_) {
    return Error::success();
  }

  std::string path = getFunctionLibraryPath();

  // Try to open and load the library.
  void *handle = dlopen(path.data(), RTLD_NOW);
  const char *dlerr = dlerror();
  RETURN_ERR_IF_NOT(handle != nullptr,
                    "Cannot open the Function Library at path '" + path +
                        "', got error: " + dlerr);
  // clear dlerror.
  dlerr = dlerror();

  // Find the "customOpVerify" function.
  customOpVerify_t verifyFunction =
      (customOpVerify_t)dlsym(handle, "customOpVerify");
  dlerr = dlerror();
  RETURN_ERR_IF_NOT(verifyFunction != nullptr,
                    (dlerr == nullptr)
                        ? "customOpVerify is nullptr in Function Library at " +
                              path
                        : "Could not find the customOpVerify symbol in the "
                          "Function Library at '" +
                              path + "', got error: " + dlerr);

  // Find the "customOpSelectImpl" function.
  customOpSelectImpl_t implSelectionFunction =
      (customOpSelectImpl_t)dlsym(handle, "customOpSelectImpl");
  dlerr = dlerror();
  RETURN_ERR_IF_NOT(
      implSelectionFunction != nullptr,
      (dlerr == nullptr)
          ? "customOpSelectImpl is nullptr in Function Library at " + path
          : "Could not find the customOpSelectImpl symbol in the Function "
            "Library at '" +
                path + "', got error: " + dlerr);

  // Find the "customOpInferShape" function.
  customOpInferShape_t shapeInfFunction =
      (customOpInferShape_t)dlsym(handle, "customOpInferShape");
  dlerr = dlerror();
  RETURN_ERR_IF_NOT(
      shapeInfFunction != nullptr,
      (dlerr == nullptr)
          ? "customOpInferShape is nullptr in Function Library at " + path
          : "Could not find the customOpInferShape symbol in the Function "
            "Library at '" +
                path + "', got error: " + dlerr);

  // set FunctionInfo.
  functionInfo_.path_ = path;
  functionInfo_.handle_ = handle;
  functionInfo_.isLoaded_ = true;
  functionInfo_.verify_ = verifyFunction;
  functionInfo_.selectImpl_ = implSelectionFunction;
  functionInfo_.inferShape_ = shapeInfFunction;
  return Error::success();
};

// Returns the verification function for the custom node registered
// via the verification function library.
// Tries to load the library if not already loaded.
// Returns nullptr in case of error while loading the library.
customOpVerify_t ProcessedOperationInfo::getVerificationFunction() {
  if (!functionInfo_.isLoaded_) {
    auto err = ERR_TO_BOOL(loadFunctionLibrary());
    if (err)
      return nullptr;
  }

  return functionInfo_.verify_;
}

// Returns ptr to shape inference function.
// Tries to load the library if not already loaded.
// Returns nullptr in case an error is encountered while loading the library.
customOpInferShape_t ProcessedOperationInfo::getShapeInferenceFunction() {
  if (!functionInfo_.isLoaded_) {
    auto err = ERR_TO_BOOL(loadFunctionLibrary());
    if (err)
      return nullptr;
  }

  return functionInfo_.inferShape_;
}

Error ProcessedOperationInfo::registerImplementations() {
  RETURN_IF_ERR(loadFunctionLibrary());
  std::string type, package;
  type = getTypeName();
  package = getPackageName();
  llvm::ArrayRef<ImplementationInfo> impls = getImplementations();

  // Check if the implementations are for a supported backend.
  // and that the backend has registered a BackendOpRepository.
  // Segregate implementations for a backend.
  std::map<std::string, std::vector<ImplementationInfo>> backendImplsMap;
  for (const auto &impl : impls) {
    auto backendName = impl.getBackendName();
    std::unique_ptr<Backend> backend(
        FactoryRegistry<std::string, Backend>::get(backendName));
    RETURN_ERR_IF_NOT(
        backend != nullptr,
        "Backend " + backendName +
            " is not registered. Cannot register OperationInfo with type " +
            type + " and package name " + package);

    BackendOpRepository *backendOpRepo = getBackendOpRepository(backendName);
    RETURN_ERR_IF_NOT(
        backendOpRepo != nullptr,
        "BackendOpRepository for backend " + backendName +
            " is not registered. Cannot register OperationInfo with type " +
            type + " and package name " + package);

    // Store the implementation in the map.
    backendImplsMap[backendName].emplace_back(impl);
  }

  for (const auto &itr : backendImplsMap) {
    BackendOpRepository *backendOpRepo = getBackendOpRepository(itr.first);
    RETURN_IF_ERR(backendOpRepo->registerImplementations(
        type, package, itr.second, functionInfo_.selectImpl_));
  }

  return Error::success();
}
} // namespace glow
