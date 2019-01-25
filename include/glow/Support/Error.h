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
    OS << "file: " << fileName_ << " line: " << lineNumber_;
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
};

/// Unwraps the T from within an llvm::Expected<T>. If the Expected<T> contains
/// an error, the program will exit.
#define EXIT_ON_ERR(...) (exitOnErr(__VA_ARGS__))

/// A temporary placeholder for EXIT_ON_ERR. This should be used only during
/// refactoring to temporarily place an EXIT_ON_ERR and should eventually be
/// replaced with either an actual EXIT_ON_ERR or code that will propogate
/// potential errors up the stack.
#define TEMP_EXIT_ON_ERR(...) (EXIT_ON_ERR(__VA_ARGS__))

/// Make a new llvm::StringError.
#define MAKE_ERR(...) llvm::make_error<GlowErr>(__FILE__, __LINE__, __VA_ARGS__)

/// Makes a new llvm::StringError and returns it.
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

/// Takes a predicate \p and if it is false then creates a new llvm::StringError
/// and returns it.
#define RETURN_ERR_IF_NOT(p, ...)                                              \
  do {                                                                         \
    if (!(p)) {                                                                \
      RETURN_ERR(__VA_ARGS__);                                                 \
    }                                                                          \
  } while (0)
} // end namespace glow

#endif // GLOW_SUPPORT_ERROR_H
