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

#include "glow/Backends/Interpreter/InterpreterOpRepository.h"
#include "glow/Backends/Interpreter/Interpreter.h"
#include "glow/Graph/OpRepository.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Error.h"

#include <dlfcn.h>
#include <sstream>
#include <string>
#include <vector>

using namespace glow;

// Loads the shared library with functions if not already loaded.
// Looks for the symbol for execution kernel and stores it.
Error InterpreterImplInfo::loadImplLibrary() {
  handle_ = dlopen(path_.data(), RTLD_NOW);
  const char *dlerr = dlerror();
  RETURN_ERR_IF_NOT(handle_ != nullptr,
                    "Cannot open the Function Library at path '" + path_ +
                        "', got error: " + dlerr);
  // clear dlerror.
  dlerr = dlerror();

  // dlsym the symbol based on selected ID.
  kernel_ = (customOpInterpreterKernel_t)dlsym(handle_, getType().data());
  dlerr = dlerror();
  RETURN_ERR_IF_NOT(kernel_ != nullptr,
                    (dlerr == nullptr)
                        ? "Interpreter kernel symbol '" + getType() +
                              " is nullptr in library at " + path_
                        : "Could not find the interpreter kernel symbol '" +
                              getType() + " in the library at '" + path_ +
                              "', got error: " + dlerr);
  isLoaded_ = true;
  return Error::success();
}

// TODO see when to call this to release library.
void InterpreterImplInfo::releaseImplLibrary() {
  if (isLoaded_) {
    int status = dlclose(handle_);
    const char *dlerr = dlerror();
    if (status != 0) {
      LOG(ERROR) << "Error in dlclose the library " << path_ << ". Got error "
                 << dlerr << std::endl;
    }
  }
}

// Utility function to call selection function and get implementation id.
Expected<customOpSelectImpl_t>
InterpreterOpRepository::getSelectionFunction(llvm::StringRef opTypeName,
                                              llvm::StringRef opPackageName) {
  std::string selKey = opPackageName.str() + "::" + opTypeName.str();
  auto itr = selectionFuncsMap_.find(selKey);

  RETURN_ERR_IF_NOT(itr != selectionFuncsMap_.end(),
                    "Could not find selection function for opType " + selKey);

  return itr->second;
}

void InterpreterOpRepository::clear() {
  // Release library.
  for (auto &kv : implementationMap_) {
    auto &implInfo = kv.second;
    implInfo.releaseImplLibrary();
  }

  // Clear maps.
  implementationMap_.clear();
  selectionFuncsMap_.clear();

  return;
}

Error InterpreterOpRepository::registerImplementations(
    llvm::StringRef opTypeName, llvm::StringRef opPackageName,
    llvm::ArrayRef<ImplementationInfo> impls, customOpSelectImpl_t implSelect) {

  RETURN_ERR_IF_NOT(implSelect != nullptr,
                    "Selection function is nullptr for opType " +
                        opTypeName.str() + " and package " +
                        opPackageName.str());

  std::string selKey = opPackageName.str() + "::" + opTypeName.str();
  auto ret = selectionFuncsMap_.insert({selKey, implSelect});
  RETURN_ERR_IF_NOT(ret.second, "Multiple registeration. Already registred "
                                "implementations for opType " +
                                    selKey);

  for (const auto &impl : impls) {
    std::string implKey = selKey + "::" + impl.getType();
    InterpreterImplInfo interpreterInfo(impl);
    RETURN_IF_ERR(interpreterInfo.loadImplLibrary());
    auto ret = implementationMap_.insert({implKey, interpreterInfo});
    RETURN_ERR_IF_NOT(ret.second, "Multiple registeration. Implementation ID " +
                                      impl.getType() +
                                      " already registred for opType " +
                                      implKey);
  }

  return Error::success();
}

Expected<customOpInterpreterKernel_t>
InterpreterOpRepository::getImplementation(llvm::StringRef opTypeName,
                                           llvm::StringRef opPackageName,
                                           llvm::StringRef flavourId) {
  std::string key =
      opPackageName.str() + "::" + opTypeName.str() + "::" + flavourId.str();

  // Return the impl for ImplId.
  auto itr = implementationMap_.find(key);
  RETURN_ERR_IF_NOT(itr != implementationMap_.end(),
                    "Implementation not found for opType " + key);

  return itr->second.getKernel();
}

REGISTER_GLOW_BACKEND_OP_REPOSITORY_FACTORY(InterpreterOpFactory,
                                            InterpreterOpRepository);
