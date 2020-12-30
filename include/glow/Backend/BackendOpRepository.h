/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

#ifndef GLOW_BACKENDS_BACKENDOPREPOSITORY_H
#define GLOW_BACKENDS_BACKENDOPREPOSITORY_H

#include "glow/Base/Traits.h"
#include "glow/Base/Type.h"
#include "glow/CustomOp/CustomOpFunctions.h"
#include "glow/Graph/OperationInfo.h"
#include "glow/Support/Error.h"
#include "glow/Support/Register.h"

namespace glow {

/// Interface for BackendOpRepository.
/// The Implementation of this interface should have the same name as the
/// backend.
class BackendOpRepository : public Named {
public:
  BackendOpRepository() : Named("") {}

  virtual ~BackendOpRepository() = default;

  // Returns the name of the BackendOpRepository.
  // Name must be same as the corresponding Backend.
  virtual std::string getBackendOpRepositoryName() const = 0;

  // Register Operation Implementations and Selection Function.
  // The ImplementationInfo \p impls can be processed and stored in any form
  // desirable by the backend. It is recommended to do some initial sanity check
  // before accepting the ImplementationInfo to communicate failure early. \p
  // implSelect can be used to decide which implementation to be used depending
  // on the input types, output types and op parameters.
  virtual Error
  registerImplementations(llvm::StringRef opTypeName,
                          llvm::StringRef opPackageName,
                          llvm::ArrayRef<ImplementationInfo> impls,
                          customOpSelectImpl_t implSelect) = 0;

  // Clears all the implementations that were registered.
  virtual void clear() = 0;

  // Backends call their BackendOpRepositories to retrieve the implementation to
  // be used. The interaction between the Backend and BackendOpRepository is not
  // defined by this interface in order to provide flexibility for indiviual
  // backends to define it in a manner that works for them.
};

/// Perform BackendOpRepository Factory registration.
/// This macro can be used by the backends to register a factory for their
/// OpRepositories.
#define REGISTER_GLOW_BACKEND_OP_REPOSITORY_FACTORY(FactoryName,               \
                                                    BackendOpRepoClass)        \
  class FactoryName : public BaseFactory<std::string, BackendOpRepository> {   \
  public:                                                                      \
    BackendOpRepository *create() override {                                   \
      if (backendOpRepo_ == nullptr) {                                         \
        backendOpRepo_ = std::make_shared<BackendOpRepoClass>();               \
      }                                                                        \
      return backendOpRepo_.get();                                             \
    }                                                                          \
    std::string getRegistrationKey() const override {                          \
      return BackendOpRepoClass::getName();                                    \
    }                                                                          \
    /*Not valid for BackendOpRepository */                                     \
    unsigned numDevices() const override { return 0; }                         \
                                                                               \
  private:                                                                     \
    std::shared_ptr<BackendOpRepoClass> backendOpRepo_;                        \
  };                                                                           \
  static RegisterFactory<std::string, FactoryName, BackendOpRepository>        \
      FactoryName##_REGISTERED;

/// Get a Backend Op Repository based on its registered name \p name.
/// returns a nullptr if the BackendOpRepository with \p name is not registered.
BackendOpRepository *getBackendOpRepository(llvm::StringRef name);

// Get available backend op repositories.
std::vector<std::string> getAvailableBackendOpRepositories();

} // namespace glow

#endif // GLOW_BACKENDS_BACKENDOPREPOSITORY_H
