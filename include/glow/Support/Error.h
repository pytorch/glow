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

#include <mutex>
#include <type_traits>

#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

#include <glog/logging.h>

namespace glow {
/// NOTE This should not be used directly, instead use EXIT_ON_ERR or
/// TEMP_EXIT_ON_ERR. Callable that takes an llvm::Error or llvm::Expected<T>
/// and exits the program if the Error is not equivalent llvm::Error::success()
/// or the Expected<T> contains an error that is not equivalent
/// llvm::Error::success()
/// TODO: replace this with a function that will print file and
/// line numbers also.
extern llvm::ExitOnError exitOnErr;

/// Is true_type only if applied to llvm::Error or a descendant.
template <typename T>
struct IsLLVMError : public std::is_base_of<llvm::Error, T> {};

/// Is true_type only if applied to llvm::Expected.
template <typename> struct IsLLVMExpected : public std::false_type {};
template <typename T>
struct IsLLVMExpected<llvm::Expected<T>> : public std::true_type {};

/// Represents errors in Glow. GlowErr track the file name and line number of
/// where they were created as well as a textual message and/or a error code to
/// help identify the type of error the occurred programtically.
class GlowErr final : public llvm::ErrorInfo<GlowErr> {
public:
  /// Used by ErrorInfo::classID.
  static const uint8_t ID;
  /// An enumeration of error codes representing various possible errors that
  /// could occur.
  /// NOTE: when updating this enum, also update ErrorCodeToString function
  /// below.
  enum class ErrorCode {
    // An unknown error ocurred. This is the default value.
    UNKNOWN,
    // Model loader encountered an unsupported shape.
    MODEL_LOADER_UNSUPPORTED_SHAPE,
    // Model loader encountered an unsupported operator.
    MODEL_LOADER_UNSUPPORTED_OPERATOR,
    // Model loader encountered an unsupported attribute.
    MODEL_LOADER_UNSUPPORTED_ATTRIBUTE,
    // Model loader encountered an unsupported datatype.
    MODEL_LOADER_UNSUPPORTED_DATATYPE,
    // Model loader encountered an unsupported ONNX version.
    MODEL_LOADER_UNSUPPORTED_ONNX_VERSION,
    // Model loader encountered an invalid protobuf.
    MODEL_LOADER_INVALID_PROTOBUF,
    // Runtime error, out of device memory.
    RUNTIME_ERROR,
    // Runtime error, out of device memory.
    RUNTIME_OUT_OF_DEVICE_MEMORY,
    // Runtime error, could not find the specified model network.
    RUNTIME_NET_NOT_FOUND,
    // Runtime error, runtime refusing to service request.
    RUNTIME_REQUEST_REFUSED,
    // Runtime error, device wasn't found.
    RUNTIME_DEVICE_NOT_FOUND,
    // Runtime error, network busy to perform any operation on it.
    RUNTIME_NET_BUSY,
    // Compilation error; node unsupported after optimizations.
    COMPILE_UNSUPPORTED_NODE_AFTER_OPTIMIZE,
    // Compilation error; Compilation context not correctly setup.
    COMPILE_CONTEXT_MALFORMED,
  };

  /// GlowErr is not convertable to std::error_code. This is included for
  /// compatiblity with ErrorInfo.
  virtual std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }

  /// Log to \p OS relevant error information including the file name and
  /// line number the GlowErr was created on as well as the message and/or error
  /// code the GlowErr was created with.
  void log(llvm::raw_ostream &OS) const override {
    OS << "location: " << fileName_ << ":" << lineNumber_;
    if (ec_ != ErrorCode::UNKNOWN) {
      OS << " error code: " << errorCodeToString(ec_);
    }
    if (!message_.empty()) {
      OS << " message: " << message_;
    }
  }

  GlowErr(llvm::StringRef fileName, size_t lineNumber, llvm::StringRef message,
          ErrorCode ec)
      : lineNumber_(lineNumber), fileName_(fileName), message_(message),
        ec_(ec) {}

  GlowErr(llvm::StringRef fileName, size_t lineNumber, ErrorCode ec,
          llvm::StringRef message)
      : lineNumber_(lineNumber), fileName_(fileName), message_(message),
        ec_(ec) {}

  GlowErr(llvm::StringRef fileName, size_t lineNumber, ErrorCode ec)
      : lineNumber_(lineNumber), fileName_(fileName), ec_(ec) {}

  GlowErr(llvm::StringRef fileName, size_t lineNumber, llvm::StringRef message)
      : lineNumber_(lineNumber), fileName_(fileName), message_(message) {}

private:
  /// Convert ErrorCode values to string.
  static std::string errorCodeToString(const ErrorCode &ec) {
    switch (ec) {
    case ErrorCode::UNKNOWN:
      return "UNKNOWN";
    case ErrorCode::MODEL_LOADER_UNSUPPORTED_SHAPE:
      return "MODEL_LOADER_UNSUPPORTED_SHAPE";
    case ErrorCode::MODEL_LOADER_UNSUPPORTED_OPERATOR:
      return "MODEL_LOADER_UNSUPPORTED_OPERATOR";
    case ErrorCode::MODEL_LOADER_UNSUPPORTED_ATTRIBUTE:
      return "MODEL_LOADER_UNSUPPORTED_ATTRIBUTE";
    case ErrorCode::MODEL_LOADER_UNSUPPORTED_DATATYPE:
      return "MODEL_LOADER_UNSUPPORTED_DATATYPE";
    case ErrorCode::MODEL_LOADER_UNSUPPORTED_ONNX_VERSION:
      return "MODEL_LOADER_UNSUPPORTED_ONNX_VERSION";
    case ErrorCode::MODEL_LOADER_INVALID_PROTOBUF:
      return "MODEL_LOADER_INVALID_PROTOBUF";
    case ErrorCode::RUNTIME_ERROR:
      return "RUNTIME_ERROR";
    case ErrorCode::RUNTIME_OUT_OF_DEVICE_MEMORY:
      return "RUNTIME_OUT_OF_DEVICE_MEMORY";
    case ErrorCode::RUNTIME_NET_NOT_FOUND:
      return "RUNTIME_NET_NOT_FOUND";
    case ErrorCode::RUNTIME_REQUEST_REFUSED:
      return "RUNTIME_REQUEST_REFUSED";
    case ErrorCode::RUNTIME_DEVICE_NOT_FOUND:
      return "RUNTIME_DEVICE_NOT_FOUND";
    case ErrorCode::RUNTIME_NET_BUSY:
      return "RUNTIME_NET_BUSY";
    case ErrorCode::COMPILE_UNSUPPORTED_NODE_AFTER_OPTIMIZE:
      return "COMPILE_UNSUPPORTED_NODE_AFTER_OPTIMIZE";
    case ErrorCode::COMPILE_CONTEXT_MALFORMED:
      return "COMPILE_CONTEXT_MALFORMED";
    };

    llvm_unreachable("unsupported ErrorCode");
  }

  /// The line number the error was generated on.
  size_t lineNumber_;
  /// The name of the file the error was generated in.
  std::string fileName_;
  /// Optional message associated with the error.
  std::string message_;
  /// Optional error code associated with the error.
  ErrorCode ec_ = ErrorCode::UNKNOWN;
}; // namespace glow

/// Unwraps the T from within an llvm::Expected<T>. If the Expected<T> contains
/// an error, the program will exit.
#define EXIT_ON_ERR(...) (exitOnErr(__VA_ARGS__))

/// A temporary placeholder for EXIT_ON_ERR. This should be used only during
/// refactoring to temporarily place an EXIT_ON_ERR and should eventually be
/// replaced with either an actual EXIT_ON_ERR or code that will propogate
/// potential errors up the stack.
#define TEMP_EXIT_ON_ERR(...) (EXIT_ON_ERR(__VA_ARGS__))

/// Make a new GlowErr.
#define MAKE_ERR(...) llvm::make_error<GlowErr>(__FILE__, __LINE__, __VA_ARGS__)

/// Makes a new GlowErr and returns it.
#define RETURN_ERR(...)                                                        \
  do {                                                                         \
    return MAKE_ERR(__VA_ARGS__);                                              \
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
      static_assert(IsLLVMError<decltype(errV)>::value,                        \
                    "Expected value to be a llvm::Error");                     \
      return std::forward<llvm::Error>(errV);                                  \
    }                                                                          \
  } while (0)

/// Takes a predicate \p and if it is false then creates a new GlowErr
/// and returns it.
#define RETURN_ERR_IF_NOT(p, ...)                                              \
  do {                                                                         \
    if (!(p)) {                                                                \
      RETURN_ERR(__VA_ARGS__);                                                 \
    }                                                                          \
  } while (0)

/// Marks the given llvm::Error as checked as long as it's value is equal to
/// llvm::Error::success(). This macro should be used as little as possible but
/// but is useful for example for creating dummy Errors that can be passed into
/// fallible constructor by reference to be filled in the event an Error occurs.
#define MARK_ERR_CHECKED(err)                                                  \
  do {                                                                         \
    bool success = !(err);                                                     \
    (void)success;                                                             \
    assert(success && "MARK_ERR_CHECKED should not be called on an "           \
                      "llvm::Error that contains an actual error.");           \
  } while (0)

/// Marks the Error \p err as as checked. \returns true if it contains an
/// error value and prints the message in the error value, returns false
/// otherwise.
inline bool errToBool(llvm::Error err) {
  if (static_cast<bool>(err)) {
    LOG(ERROR) << "Converting error to boolean: "
               << llvm::toString(std::move(err));
    return true;
  }
  return false;
}

template <typename T> llvm::Error takeErr(llvm::Expected<T> e) {
  if (!bool(e)) {
    return e.takeError();
  } else {
    return llvm::Error::success();
  }
}

/// This class holds an llvm::Error provided via the add method. If an Error is
/// added when the class already holds an Error, it will discard the new Error
/// in favor of the original one. All methods in OneErrOnly are thread-safe.
class OneErrOnly {
  llvm::Error err_ = llvm::Error::success();
  std::mutex m_;

public:
  /// Add a new llvm::Error \p err to be stored. If an existing Error has
  /// already been added then the contents of the new error will be logged and
  /// the new err will be discarded. \returns true if \p err was stored and
  /// \returns false otherwise. If \p err is an empty Error then does nothing
  /// and \returns false;
  bool set(llvm::Error err);

  /// \returns the stored llvm:Error clearing out the storage of the class.
  llvm::Error get();

  /// \returns true if contains an Error and false otherwise.
  bool containsErr();
};

} // end namespace glow

#endif // GLOW_SUPPORT_ERROR_H
