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
namespace detail {
/// NOTE This should not be used directly, instead use UNWRAP or TEMP_UNWRAP.
/// Callable that takes an llvm::Error or llvm::Expected<T> and exits the
/// program if the Error is not equivalent llvm::Error::success() or the
/// Expected<T> contains an error that is not equivalent llvm::Error::success()
/// TODO(jackmontgomery) replace this with a function that will print file and
/// line numbers also.
extern llvm::ExitOnError exitOnErr;

/// Take a message \p str and prepend it with the given \p file and \p line
/// number. This is useful for augmenting StringErrors with information about
/// where they were generated.
std::string addFileAndLineToError(llvm::StringRef str, llvm::StringRef file,
                                  uint32_t line);
} // namespace detail

/// Is true_type only if applied to llvm::Error or a descendant.
template <typename T>
struct IsLLVMError : public std::is_base_of<llvm::Error, T> {};

/// Is true_type only if applied to llvm::Expected.
template <typename> struct IsLLVMExpected : public std::false_type {};
template <typename T>
struct IsLLVMExpected<llvm::Expected<T>> : public std::true_type {};

/// Unwrap the T from within an llvm::Expected<T>. If the Expected<T> contains
/// an error, the program will abort.
#define UNWRAP(...) (detail::exitOnErr(__VA_ARGS__))

/// A temporary placeholder for UNWRAP. This should be used only during
/// refactoring to temporarily place an UNWRAP and should eventually be
/// replaced with either an actual UNWRAP or code that will propogate potential
/// errors up the stack.
#define TEMP_UNWRAP(...) (UNWRAP(__VA_ARGS__))

/// Make a new llvm::StringError.
#define MAKE_ERR(str)                                                          \
  llvm::make_error<llvm::StringError>(                                         \
      (detail::addFileAndLineToError(str, __FILE__, __LINE__)),                \
      llvm::inconvertibleErrorCode())

/// Makes a new llvm::StringError and returns it.
#define RETURN_ERR(str)                                                        \
  do {                                                                         \
    return MAKE_ERR(str);                                                      \
  } while (0)

/// Returns llvm::Error::success().
#define RETURN_SUCCESS()                                                       \
  do {                                                                         \
    return llvm::Error::success();                                             \
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
// TODO(jackmontgomery) extend this to work with llvm::Expected as well.
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
