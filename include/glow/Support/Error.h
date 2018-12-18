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
#ifndef GLOW_SUPPORT_ERROR_H
#define GLOW_SUPPORT_ERROR_H

#include <type_traits>

#include "llvm/Support/Error.h"

namespace glow {
/// NOTE This should not be used directly, instead use EXIT_ON_ERR or
/// TEMP_EXIT_ON_ERR. Callable that takes an llvm::Error or llvm::Expected<T>
/// and exits the program if the Error is not equivalent llvm::Error::success()
/// or the Expected<T> contains an error that is not equivalent
/// llvm::Error::success()
/// TODO: replace this with a function that will print file and
/// line numbers also.
extern llvm::ExitOnError exitOnErr;

/// Take a message \p str and prepend it with the given \p file and \p line
/// number. This is useful for augmenting StringErrors with information about
/// where they were generated.
std::string addFileAndLineToError(llvm::StringRef str, llvm::StringRef file,
                                  uint32_t line);

/// Is true_type only if applied to llvm::Error or a descendant.
template <typename T>
struct IsLLVMError : public std::is_base_of<llvm::Error, T> {};

/// Is true_type only if applied to llvm::Expected.
template <typename> struct IsLLVMExpected : public std::false_type {};
template <typename T>
struct IsLLVMExpected<llvm::Expected<T>> : public std::true_type {};

template <typename ThisErrT> class BaseErr : public llvm::ErrorInfoBase {
public:
  static const void *classID() { return &ThisErrT::ID; }

  const void *dynamicClassID() const override { return &ThisErrT::ID; }

  bool isA(const void *const ClassID) const override {
    return ClassID == classID() || llvm::ErrorInfoBase::isA(ClassID);
  }

  virtual std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }

  void log(llvm::raw_ostream &OS) const override {
    OS << "file: " << fileName_ << " line: " << lineNumber_ << " error: \"";
    static_cast<const ThisErrT *>(this)->logErr(OS);
    OS << "\"";
  }

protected:
  BaseErr(llvm::StringRef fileName, int32_t lineNumber) {
    fileName_ = fileName;
    lineNumber_ = lineNumber;
  }

private:
  int32_t lineNumber_;
  std::string fileName_;
};

class ModelLoadingErr final : public BaseErr<ModelLoadingErr> {
public:
  // Used by BaseErr.
  static char ID;

  enum EC {
    UNKNOWN = 0,
    UNSUPPORTED_SHAPE = 1,
    UNSUPPORTED_OPERATOR = 2,
    UNSUPPORTED_ATTRIBUTE = 3,
    UNSUPPORTED_DATATYPE = 4,
  };

  static llvm::StringRef ecToString(const EC ec) {
    switch (ec) {
    case UNKNOWN:
      return "UNKNOWN";
    case UNSUPPORTED_SHAPE:
      return "UNSUPPORTED_SHAPE";
    case UNSUPPORTED_OPERATOR:
      return "UNSUPPORTED_OPERATOR";
    case UNSUPPORTED_ATTRIBUTE:
      return "UNSUPPORTED_ATTRIBUTE";
    case UNSUPPORTED_DATATYPE:
      return "UNSUPPORTED_DATATYPE";
    }
  }

  ModelLoadingErr(llvm::StringRef fileName, int32_t lineNumber,
                  llvm::StringRef message, EC ec)
      : BaseErr(fileName, lineNumber) {
    message_ = message;
    ec_ = ec;
  }

  ModelLoadingErr(llvm::StringRef fileName, int32_t lineNumber,
                  llvm::StringRef message)
      : BaseErr(fileName, lineNumber) {
    message_ = message;
  }

  void logErr(llvm::raw_ostream &OS) const {
    OS << "Failed to load model. Encountered " << ecToString(ec_)
       << " error: " << message_;
  }

  EC getEC() const { return ec_; }

private:
  std::string message_;
  EC ec_ = EC::UNKNOWN;
};

#define MAKE_MODEL_LOADING_ERR(...)                                            \
  llvm::make_error<ModelLoadingErr>(__FILE__, __LINE__, __VA_ARGS__)

#define RETURN_MODEL_LOADING_ERR(...)                                          \
  do {                                                                         \
    return MAKE_MODEL_LOADING_ERR(__VA_ARGS__);                                \
  } while (0)

class RuntimeErr final : public BaseErr<RuntimeErr> {
public:
  // Used by BaseErr.
  static char ID;

  enum EC {
    UNKNOWN = 0,
    NO_DEVICE_MEMORY = 1,
    NO_HOST_MEMORY = 2,
    UNKNOWN_NETWORK = 3,
    QUEUE_FULL = 4,
    PARTITION_FAILED = 5,
  };

  static llvm::StringRef ecToString(const EC ec) {
    switch (ec) {
    case UNKNOWN:
      return "UNKNOWN";
    case NO_DEVICE_MEMORY:
      return "NO_DEVICE_MEMORY";
    case NO_HOST_MEMORY:
      return "NO_HOST_MEMORY";
    case UNKNOWN_NETWORK:
      return "UNKNOWN_NETWORK";
    case QUEUE_FULL:
      return "QUEUE_FULL";
    case PARTITION_FAILED:
      return "PARTITION_FAILED";
    }
  }

  RuntimeErr(llvm::StringRef fileName, int32_t lineNumber, EC ec)
      : BaseErr(fileName, lineNumber) {
    ec_ = ec;
  }

  void logErr(llvm::raw_ostream &OS) const {
    OS << "Encountered runtime error: " << ecToString(ec_);
  }

  EC getEC() const { return ec_; }

private:
  EC ec_ = EC::UNKNOWN;
};

#define MAKE_RUNTIME_ERR(...)                                                  \
  llvm::make_error<RuntimeErr>(__FILE__, __LINE__, __VA_ARGS__)

#define RETURN_RUNTIME_ERR(...)                                                \
  do {                                                                         \
    return MAKE_RUNTIME_ERR(__VA_ARGS__);                                      \
  } while (0)

/// Unwraps the T from within an llvm::Expected<T>. If the Expected<T> contains
/// an error, the program will exit.
#define EXIT_ON_ERR(...) (exitOnErr(__VA_ARGS__))

/// A temporary placeholder for EXIT_ON_ERR. This should be used only during
/// refactoring to temporarily place an EXIT_ON_ERR and should eventually be
/// replaced with either an actual EXIT_ON_ERR or code that will propogate
/// potential errors up the stack.
#define TEMP_EXIT_ON_ERR(...) (EXIT_ON_ERR(__VA_ARGS__))

/// Make a new llvm::StringError.
#define MAKE_ERR(str)                                                          \
  llvm::make_error<llvm::StringError>(                                         \
      (addFileAndLineToError(str, __FILE__, __LINE__)),                        \
      llvm::inconvertibleErrorCode())

/// Makes a new llvm::StringError and returns it.
#define RETURN_ERR(str)                                                        \
  do {                                                                         \
    return MAKE_ERR(str);                                                      \
  } while (0)

/// Takes an llvm::Expected<T> \p lhsOrErr and if it is an Error then returns
/// it, otherwise takes the value from lhsOrErr and assigns it to \p rhs.
#define ASSIGN_VALUE_OR_RETURN_ERR(rhs, lhsOrErr)                              \
  do {                                                                         \
    auto lhsOrErrV = (lhsOrErr);                                               \
    static_assert(IsLLVMExpected<decltype(lhsOrErrV)>(),                       \
                  "Expected value to be a llvm::Expected");                    \
    if (lhsOrErrV) {                                                           \
      rhs = std::move(lhsOrErrV.get());                                        \
    } else {                                                                   \
      return lhsOrErrV.takeError();                                            \
    }                                                                          \
  } while (0)

/// Takes an llvm::Error and returns it if it's not success.
// TODO: extend this to work with llvm::Expected as well.
#define RETURN_IF_ERR(err)                                                     \
  do {                                                                         \
    if (auto errV = std::forward<llvm::Error>(err)) {                          \
      static_assert(IsLLVMError<decltype(errV)>(),                             \
                    "Expected value to be a llvm::Error");                     \
      return std::forward<llvm::Error>(errV);                                  \
    }                                                                          \
  } while (0)

/// Takes a predicate \p and if it is false then creates a new llvm::StringError
/// and returns it.
#define RETURN_ERR_IF_NOT(p, str)                                              \
  do {                                                                         \
    if (!(p)) {                                                                \
      RETURN_ERR(str);                                                         \
    }                                                                          \
  } while (0)
} // end namespace glow

#endif // GLOW_SUPPORT_ERROR_H
