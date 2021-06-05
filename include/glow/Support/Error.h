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
#ifndef GLOW_SUPPORT_ERROR_H
#define GLOW_SUPPORT_ERROR_H

#include <cassert>
#include <memory>
#include <mutex>
#include <type_traits>

#include <glog/logging.h>

/// NOTE: please only use code and macros that resides outside of the detail
/// namespace in Error.h and Error.cpp so as to preserve a layer of
/// abstraction between Error/Expected types and the specific classes that
/// implement them.

/// Consumes an Error and \returns true iff the error contained an
/// ErrorValue. Calls the log method on ErrorValue if the optional argument \p
/// log is passed.
#define ERR_TO_BOOL(...)                                                       \
  (glow::detail::errorToBool(__FILE__, __LINE__, __VA_ARGS__))

/// Consumes an Error and \returns "success" if it does not contain an
/// ErrorValue or the result of calling the log() if it does.
#define ERR_TO_STRING(...)                                                     \
  (glow::detail::errorToString(__FILE__, __LINE__, __VA_ARGS__))

/// Consumes an Error. Calls the log method on the ErrorValue if the
/// optional argument \p log is passed.
#define ERR_TO_VOID(...)                                                       \
  (glow::detail::errorToVoid(__FILE__, __LINE__, __VA_ARGS__))

/// Unwraps the T from within an Expected<T>. If the Expected<T> contains
/// an ErrorValue, the program will exit.
#define EXIT_ON_ERR(...)                                                       \
  (glow::detail::exitOnError(__FILE__, __LINE__, __VA_ARGS__))

/// A temporary placeholder for EXIT_ON_ERR. This should be used only during
/// refactoring to temporarily place an EXIT_ON_ERR and should eventually be
/// replaced with either an actual EXIT_ON_ERR or code that will propogate
/// potential errors up the stack.
#define TEMP_EXIT_ON_ERR(...) (EXIT_ON_ERR(__VA_ARGS__))

/// Makes a new Error.
#define MAKE_ERR(...) glow::detail::makeError(__FILE__, __LINE__, __VA_ARGS__)

/// Takes an Expected<T> \p rhsOrErr and if it is an Error then returns
/// it, otherwise takes the value from rhsOrErr and assigns it to \p lhs.
#define ASSIGN_VALUE_OR_RETURN_ERR(lhs, rhsOrErr)                              \
  do {                                                                         \
    auto rhsOrErrV = (rhsOrErr);                                               \
    static_assert(glow::detail::IsExpected<decltype(rhsOrErrV)>(),             \
                  "Expected value to be a Expected");                          \
    if (rhsOrErrV) {                                                           \
      lhs = std::move(rhsOrErrV.get());                                        \
    } else {                                                                   \
      auto err = rhsOrErrV.takeError();                                        \
      err.addToStack(__FILE__, __LINE__);                                      \
      return std::forward<Error>(err);                                         \
    }                                                                          \
  } while (0)

/// Takes an Expected<T> \p rhsOrErr and if it is an Error then calls FAIL()
/// otherwise takes the value from rhsOrErr and assigns it to \p lhs.
#define ASSIGN_VALUE_OR_FAIL_TEST(lhs, rhsOrErr)                               \
  do {                                                                         \
    auto rhsOrErrV = (rhsOrErr);                                               \
    static_assert(glow::detail::IsExpected<decltype(rhsOrErrV)>(),             \
                  "Expected value to be a Expected");                          \
    if (rhsOrErrV) {                                                           \
      lhs = std::move(rhsOrErrV.get());                                        \
    } else {                                                                   \
      auto err = rhsOrErrV.takeError();                                        \
      FAIL() << ERR_TO_STRING(std::move(err));                                 \
    }                                                                          \
  } while (0)

/// Takes an Expected<T> \p rhsOrErr and if it is an Error then LOG(FATAL)'s,
/// otherwise takes the value from rhsOrErr and assigns it to \p lhs.
#define ASSIGN_VALUE_OR_FATAL(lhs, rhsOrErr)                                   \
  do {                                                                         \
    auto rhsOrErrV = (rhsOrErr);                                               \
    static_assert(glow::detail::IsExpected<decltype(rhsOrErrV)>(),             \
                  "Expected value to be a Expected");                          \
    if (rhsOrErrV) {                                                           \
      lhs = std::move(rhsOrErrV.get());                                        \
    } else {                                                                   \
      auto err = rhsOrErrV.takeError();                                        \
      LOG(FATAL) << ERR_TO_STRING(std::move(err));                             \
    }                                                                          \
  } while (0)

/// Takes an Error, adds stack information to it, and returns it unconditionally
#define RETURN_ERR(err)                                                        \
  do {                                                                         \
    auto errV = std::forward<glow::detail::GlowError>(err);                    \
    static_assert(glow::detail::IsError<decltype(errV)>::value,                \
                  "Expected value to be a Error");                             \
    errV.addToStack(__FILE__, __LINE__);                                       \
    return std::forward<Error>(errV);                                          \
  } while (0)

/// Takes an Error and returns it if it's not success.
#define RETURN_IF_ERR(err)                                                     \
  do {                                                                         \
    if (auto errV = std::forward<glow::detail::GlowError>(err)) {              \
      static_assert(glow::detail::IsError<decltype(errV)>::value,              \
                    "Expected value to be a Error");                           \
      errV.addToStack(__FILE__, __LINE__);                                     \
      return std::forward<Error>(errV);                                        \
    }                                                                          \
  } while (0)

/// Takes an Expected and returns it if it's not success.
#define RETURN_IF_EXPECTED_IS_ERR(expInp)                                      \
  do {                                                                         \
    auto expV = (expInp);                                                      \
    static_assert(glow::detail::IsExpected<decltype(expV)>(),                  \
                  "Expected value to be a Expected");                          \
    RETURN_IF_ERR((expV).takeError());                                         \
  } while (0)

/// Takes an Error and if it contains an ErrorValue then calls FAIL().
#define FAIL_TEST_IF_ERR(err)                                                  \
  do {                                                                         \
    if (auto errV = std::forward<glow::detail::GlowError>(err)) {              \
      static_assert(glow::detail::IsError<decltype(errV)>::value,              \
                    "Expected value to be a Error");                           \
      FAIL() << ERR_TO_STRING(std::move(errV));                                \
    }                                                                          \
  } while (0)

/// Takes a predicate \p and if it is false then creates a new Error
/// and returns it.
#define RETURN_ERR_IF_NOT(p, ...)                                              \
  do {                                                                         \
    if (!(p)) {                                                                \
      return MAKE_ERR(__VA_ARGS__);                                            \
    }                                                                          \
  } while (0)

/// Takes an Error if it's not success then adds the given message to the stack.
#define ADD_MESSAGE_TO_ERR_STACK(err, msg)                                     \
  (err).addToStack(__FILE__, __LINE__, msg)

namespace glow {

/// Forward declarations.
namespace detail {
class GlowError;
class GlowErrorSuccess;
class GlowErrorEmpty;
class GlowErrorValue;
template <typename T> class GlowExpected;
} // namespace detail

/// Type aliases to decouple Error and Expected from underlying implementation.
using Error = detail::GlowError;
using ErrorSuccess = detail::GlowErrorSuccess;
using ErrorEmpty = detail::GlowErrorEmpty;
using ErrorValue = detail::GlowErrorValue;
template <typename T> using Expected = detail::GlowExpected<T>;

/// NOTE: detail namespace contains code that should not be used outside of
/// Error.h and Error.cpp. Please instead use types and macros defined above.
namespace detail {
/// enableCheckingErrors is used to enable assertions that every Error and
/// Expected has its status checked before it is destroyed. This should be
/// enabled in debug builds but turned off otherwise.
#ifdef NDEBUG
static constexpr bool enableCheckingErrors = false;
#else
static constexpr bool enableCheckingErrors = true;
#endif

/// Is true_type only if applied to Error or a descendant.
template <typename T> struct IsError : public std::is_base_of<GlowError, T> {};

/// Is true_type only if applied to Expected.
template <typename> struct IsExpected : public std::false_type {};
template <typename T>
struct IsExpected<GlowExpected<T>> : public std::true_type {};

/// CheckState<DoChecks> is a common base class for Error and Expected that
/// tracks whether their state has been checked or not if DoChecks is true
/// and otherwise it does nothing and has no members so as to not take extra
/// space. This is used to ensure that all Errors and Expecteds are checked
/// before they are destroyed.
template <bool DoChecks> class CheckState;

/// Specialization of CheckState with checking enabled.
template <> class CheckState<true> {
  /// Whether or not the a check has occurred.
  bool checked_ = false;

public:
  /// Set the state of checked.
  inline void setChecked(bool checked) { checked_ = checked; }

  /// Asserts that the state has been checked.
  inline void ensureChecked() const {
    assert(checked_ && "Unchecked Error or Expected");
  }
  CheckState() : checked_(false) {}

  /// Destructor that is used to ensure that base classes have been checked.
  ~CheckState() { ensureChecked(); }

  /// NOTE: Only use for testing!
  bool isChecked() const { return checked_; }
};

/// Specialization of CheckState with checking disabled.
template <> class CheckState<false> {
public:
  inline void setChecked(bool checked) {}
  inline void ensureChecked() const {}
  /// NOTE: Only use for testing!
  bool isChecked() const { return true; }
};

/// Opaque is an aligned opaque container for some type T. It holds a T in-situ
/// but will not destroy it automatically when the Opaque is destroyed but
/// instead only when the destroy() method is called.
template <typename T> class Opaque {
private:
  alignas(T) char payload_[sizeof(T)];

public:
  /// Sets the value within this Opaque container.
  void set(T t) { new (payload_) T(std::forward<T>(t)); }

  /// Gets the value within this Opaque container.
  T &get() { return *reinterpret_cast<T *>(payload_); }

  /// Gets the value within this Opaque container.
  const T &get() const { return *reinterpret_cast<const T *>(payload_); }

  /// Call the destructor of the value in this container.
  void destroy() { get().~T(); }
};

/// This method is the only way to destroy an Error \p error and mark it as
/// checked when it contains an ErrorValue. It \returns the contained
/// ErrorValue.
/// NOTE: This method should not be used directly, use one of the methods that
/// calls this.
std::unique_ptr<GlowErrorValue> takeErrorValue(GlowError error);

/// Takes an Error \p error and asserts that it does not contain an ErrorValue.
/// Uses \p fileName and \p lineNumber for logging.
void exitOnError(const char *fileName, size_t lineNumber, GlowError error);

/// ErrorValue contains information about an error that occurs at runtime. It is
/// not used directly but instead is passed around inside of the Error and
/// Expected containers. It should only be constructed using the makeError
/// method.
class GlowErrorValue final {
public:
  struct StackEntry {
    std::string fileName;
    size_t lineNumber;
    std::string message;

    StackEntry(const std::string &fileNameParam, size_t lineNumberParam)
        : fileName(fileNameParam), lineNumber(lineNumberParam) {}

    StackEntry(const std::string &fileNameParam, size_t lineNumberParam,
               const std::string &messageParam)
        : fileName(fileNameParam), lineNumber(lineNumberParam),
          message(messageParam) {}
  };

  /// An enumeration of error codes representing various possible errors that
  /// could occur.
  /// NOTE: when updating this enum, also update errorCodeToString function
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
    // Partitioner error.
    PARTITIONER_ERROR,
    // Runtime error, general error.
    RUNTIME_ERROR,
    // Runtime error, error while loading deferred weights.
    RUNTIME_DEFERRED_WEIGHT_ERROR,
    // Runtime error, out of device memory.
    RUNTIME_OUT_OF_DEVICE_MEMORY,
    // Runtime error, could not find the specified model network.
    RUNTIME_NET_NOT_FOUND,
    // Runtime error, runtime refusing to service request.
    RUNTIME_REQUEST_REFUSED,
    // Runtime error, device wasn't found.
    RUNTIME_DEVICE_NOT_FOUND,
    // Non-recoverable device error
    RUNTIME_DEVICE_NONRECOVERABLE,
    // Runtime error, network busy to perform any operation on it.
    RUNTIME_NET_BUSY,
    // Device error, not supported.
    DEVICE_FEATURE_NOT_SUPPORTED,
    // Compilation error; node unsupported after optimizations.
    COMPILE_UNSUPPORTED_NODE_AFTER_OPTIMIZE,
    // Compilation error; Compilation context not correctly setup.
    COMPILE_CONTEXT_MALFORMED,
    // Model writer encountered an invalid file name.
    MODEL_WRITER_INVALID_FILENAME,
    // Model writer cannot serialize graph to the file.
    MODEL_WRITER_SERIALIZATION_ERROR,
    // Compilation error; IR unsupported after generation.
    COMPILE_UNSUPPORTED_IR_AFTER_GENERATE,
    // Compilation error; IR unsupported after optimization.
    COMPILE_UNSUPPORTED_IR_AFTER_OPTIMIZE,
  };

  /// Log to \p os relevant error information including the file name and
  /// line number the ErrorValue was created on as well as the message and/or
  /// error code the ErrorValue was created with. If \p warning is true then
  /// the log message will replace "Error" with "Warning", this is useful for
  /// when Errors are used in non-exceptional conditions.
  template <typename StreamT>
  void log(StreamT &os, bool warning = false) const {
    const char *mType = warning ? "Warning" : "Error";
    if (ec_ != ErrorCode::UNKNOWN) {
      os << "\n" << mType << " code: " << errorCodeToString(ec_);
    }
    if (!message_.empty()) {
      os << "\n" << mType << " message: " << message_;
    }
    os << "\n\n" << mType << " return stack:";
    for (const auto &p : stack_) {
      os << "\n----------------------------------------------------------------"
            "----------------";
      os << "\n" << p.fileName.c_str() << ":" << p.lineNumber;
      if (!p.message.empty()) {
        os << " with message:\n";
        os << p.message.c_str();
      }
    }
    os << "\n----------------------------------------------------------------"
          "----------------";
  }

  /// Add \p filename and \p lineNumber to the ErrorValue's stack for logging.
  void addToStack(const std::string &fileName, size_t lineNumber,
                  const std::string &message = "") {
    stack_.emplace_back(fileName, lineNumber, message);
  }

  /// If \p warning is true then the log message will replace "Error" with
  /// "Warning", this is useful for when Errors are used in non-exceptional
  /// conditions.
  std::string logToString(bool warning = false) const;

  /// Return the error code.
  bool isFatalError() const {
    return ec_ == ErrorCode::RUNTIME_DEVICE_NONRECOVERABLE;
  }

  ErrorCode getErrorCode() const { return ec_; }

  GlowErrorValue(std::string message, ErrorCode ec)
      : message_(message), ec_(ec) {}

  GlowErrorValue(ErrorCode ec, std::string message)
      : message_(message), ec_(ec) {}

  GlowErrorValue(ErrorCode ec) : ec_(ec) {}

  GlowErrorValue(std::string message) : message_(message) {}

private:
  /// Convert ErrorCode values to string.
  static std::string errorCodeToString(const ErrorCode &ec);

  /// Optional message associated with the error.
  std::string message_;
  /// The (filename, line number, optional message) of all places that created,
  /// forwarded, and destroyed the ErrorValue.
  std::vector<StackEntry> stack_;
  /// Optional error code associated with the error.
  ErrorCode ec_ = ErrorCode::UNKNOWN;
};

/// Overload for operator<< for logging an ErrorValue \p errorValue to a stream
/// \p os.
template <typename StreamT>
StreamT &operator<<(StreamT &os, const GlowErrorValue &errorValue) {
  errorValue.log(os);
  return os;
}

/// Error is a container for pointers to ErrorValues. If an ErrorValue is
/// contained Error ensures that it is checked before being destroyed.
class GlowError : protected detail::CheckState<detail::enableCheckingErrors> {
  template <typename T> friend class GlowExpected;
  friend std::unique_ptr<GlowErrorValue> detail::takeErrorValue(GlowError);

  /// Pointer to ErrorValue managed by this Error. Can be null if no error
  /// occurred. Use getters and setters defined below to access this since they
  /// also will modify the CheckState.
  std::unique_ptr<GlowErrorValue> errorValue_;

  /// \return true if an ErrorValue is contained.
  inline bool hasErrorValue() const { return nullptr != errorValue_; }

  /// Sets the value of errorValue_ to \p errorValue ensuring not to overwrite
  /// any previously contained ErrorValues that were unchecked. This is skipped
  /// however if \p skipCheck is passed.
  /// NOTE: skipCheck should only be used by constructors.
  inline void setErrorValue(std::unique_ptr<GlowErrorValue> errorValue,
                            bool skipCheck = false) {
    // Can't overwrite an existing error unless we say not to check.
    if (skipCheck) {
      assert(errorValue_ == nullptr &&
             "Trying to skip state check on an Error that "
             "contains an ErrorValue is a bug because this should only happen "
             "in a constructor and then no ErrorValue should be contained.");
    } else {
      ensureChecked();
    }

    errorValue_ = std::move(errorValue);
    setChecked(false);
  }

  /// \returns the contents of errorValue_ by moving them. Marks the Error as
  /// checked no matter what.
  /// NOTE: This is the only way to mark an Error that contains an ErrorValue as
  /// checked.
  inline std::unique_ptr<GlowErrorValue> takeErrorValue() {
    setChecked(true);
    return std::move(errorValue_);
  }

#ifdef WIN32
public:
#else
protected:
#endif
  /// Construct a new empty Error.
  explicit GlowError() { setErrorValue(nullptr, /*skipCheck*/ true); }

public:
  /// Construct an Error from an ErrorValue \p errorValue.
  GlowError(std::unique_ptr<GlowErrorValue> errorValue) {
    assert(errorValue &&
           "Cannot construct an Error from a null ErrorValue ptr");
    setErrorValue(std::move(errorValue), /*skipCheck*/ true);
  }

  /// Move construct an Error from another Error \p other.
  GlowError(GlowError &&other) {
    setErrorValue(std::move(other.errorValue_), /*skipCheck*/ true);
    other.setChecked(true);
  }

  /// Construct an Error from an ErrorEmpty \p other. This is a special case
  /// constructor that will mark the constructed Error as being checked. This
  /// should only be used for creating Errors that will be passed into things
  /// like fallible constructors of other classes to be written to.
  GlowError(GlowErrorEmpty &&other);

  /// Move assign Error from another Error \p other.
  GlowError &operator=(GlowError &&other) {
    setErrorValue(std::move(other.errorValue_));
    other.setChecked(true);
    return *this;
  }

  /// Add \p filename and \p lineNumber to the contained ErrorValue's stack for
  /// logging.
  void addToStack(const std::string &fileName, size_t lineNumber,
                  const std::string &message = "") {
    if (hasErrorValue()) {
      errorValue_->addToStack(fileName, lineNumber, message);
    }
  }

  /// Create an Error not containing an ErrorValue that is signifies success
  /// instead of failure of an operation.
  /// NOTE: this Error must still be checked before being destroyed.
  static GlowErrorSuccess success();

  /// Create an empty Error that is signifies that an operation has not yet
  /// occurred. This should only be used when another Error will be assigned to
  /// this Error for example when calling a fallible constructor that takes an
  /// Error reference as a parameter.
  /// NOTE: this Error is considered to be "pre-checked" and therefore can be
  /// destroyed at any time.
  static GlowErrorEmpty empty();

  // Disable copying Errors.
  GlowError(const GlowError &) = delete;
  GlowError &operator=(const GlowError &) = delete;

  /// Overload of operator bool() that \returns true if an ErrorValue is
  /// contained.
  /// NOTE: This marks the Error as checked only if no ErrorValue is contained.
  /// If an ErrorValue is contained then that ErrorValue must be handled in
  /// order to mark as checked.
  explicit operator bool() {
    // Only mark as checked when there isn't an ErrorValue contained.
    bool hasError = hasErrorValue();
    if (!hasError) {
      setChecked(true);
    }
    return hasError;
  }

  /// \returns a pointer to the contained ErrorValue or nullptr if no
  /// ErrorValue is contained in this Error.
  inline const GlowErrorValue *peekErrorValue() const {
    return errorValue_.get();
  }

  /// NOTE: Only use for testing!
  bool isChecked_() const { return isChecked(); }
};

/// ErrorSuccess is a special Error that is used to mark the absents of an
/// error. It shouldn't be constructed directly but instead using
/// Error::success().
class GlowErrorSuccess final : public GlowError {};

/// See declaration in Error for details.
inline GlowErrorSuccess GlowError::success() { return GlowErrorSuccess(); }

/// ErrorSuccess is a special Error that is used to contain the future state of
/// a fallible process that hasn't yet occurred. It shouldn't be
/// constructed directly but instead using Error::empty(). See comments on
/// Error::empty() method for more details.
class GlowErrorEmpty final : public GlowError {};

/// See declaration in Error for details.
inline GlowErrorEmpty GlowError::empty() { return GlowErrorEmpty(); }

template <typename T> class GlowExpected;
template <typename T>
T exitOnError(const char *fileName, size_t lineNumber,
              GlowExpected<T> expected);

/// Expected is a templated container for either a value of type T or an
/// ErrorValue. It is used for fallible processes that may return either a value
/// or encounter an error. Expected ensures that its state has been checked for
/// errors before destruction.
template <typename T>
class GlowExpected final
    : protected detail::CheckState<detail::enableCheckingErrors> {
  friend T detail::exitOnError<>(const char *fileName, size_t lineNumber,
                                 GlowExpected<T> expected);
  template <typename OtherT> friend class GlowExpected;

  /// Union type between ErrorValue and T. Holds both in Opaque containers so
  /// that lifetime management is manual and tied to the lifetime of Expected.
  union Payload {
    detail::Opaque<std::unique_ptr<GlowErrorValue>> asErrorValue;
    detail::Opaque<T> asValue;
  };

  /// A union that contains either an ErrorValue if an error occurred
  /// or a value of type T.
  Payload payload_;

  /// Whether or not payload_ contains an Error. Expected cannot be constructed
  /// from ErrorSuccess so if an ErrorValue is contained it is a legitimate
  /// Error.
  bool isError_;

  /// Getter for isError_.
  inline bool getIsError() const { return isError_; }

  /// Setter for isError_.
  inline void setIsError(bool isError) { isError_ = isError; }

  /// Asserts that an ErrorValue is contained not a Value.
  inline void ensureError() {
    assert(getIsError() && "Trying to get an ErrorValue of an Expected that "
                           "doesn't contain an ErrorValue");
  }

  /// Asserts that a Value is contained not an ErrorValue
  inline void ensureValue() {
    assert(
        !getIsError() &&
        "Trying to get a Value of an Expected that doesn't contain an Value");
  }

  /// Setter for payload_ that inserts an ErrorValue \p errorValue. If \p
  /// skipCheck is true then don't check that the current payload has been
  /// checked before setting otherwise do check.
  /// NOTE: Only constructors of Expected should use skipCheck.
  inline void setErrorValue(std::unique_ptr<GlowErrorValue> errorValue,
                            bool skipCheck = false) {
    if (!skipCheck) {
      ensureChecked();
    }
    setIsError(true);
    return payload_.asErrorValue.set(std::move(errorValue));
  }

  /// Getter for payload_ to retrieve an ErrorValue. Ensures that an ErrorValue
  /// is contained and that it has been checked.
  inline GlowErrorValue *getErrorValue() {
    ensureError();
    ensureChecked();
    return payload_.asErrorValue.get().get();
  }

  /// Getter for payload_ to retrieve an ErrorValue. Ensures that an ErrorValue
  /// is contained and that it has been checked.
  inline const GlowErrorValue *getErrorValue() const {
    ensureError();
    ensureChecked();
    return payload_.asErrorValue.get().get();
  }

  /// \returns the ErrorValue contents of payload_ by moving them. Marks the
  /// Expected as checked no matter what.
  /// NOTE: This is the only way to mark an Expected that contains an ErrorValue
  /// as checked.
  inline std::unique_ptr<GlowErrorValue> takeErrorValue() {
    ensureError();
    setChecked(true);
    return std::move(payload_.asErrorValue.get());
  }

  /// Sets payload_ with a value of type T \p value. If \p skipCheck is true
  /// then don't check that the current payload has been checked before setting
  /// otherwise do check.
  /// NOTE: Only constructors of Expected should use skipCheck.
  inline void setValue(T value, bool skipCheck = false) {
    static_assert(!std::is_reference<T>::value,
                  "Expected has not been equipped to hold references yet.");

    if (!skipCheck) {
      ensureChecked();
    }
    setIsError(false);
    return payload_.asValue.set(std::move(value));
  }

  /// \returns a value T contained in payload_. Ensures that value is contained
  /// by payload_ and that it has been checked already.
  inline T *getValue() {
    ensureValue();
    ensureChecked();
    return &payload_.asValue.get();
  }

  /// \returns a value T contained in payload_. Ensures that value is contained
  /// by payload_ and that it has been checked already.
  inline const T *getValue() const {
    ensureValue();
    ensureChecked();
    return &payload_.asValue.get();
  }

  /// \returns the value contents of payload_ by moving them. Marks the Expected
  /// as checked no matter what.
  inline T takeValue() {
    ensureValue();
    setChecked(true);
    return std::move(payload_.asValue.get());
  }

  /// Add \p filename and \p lineNumber to the contained ErrorValue's stack for
  /// logging.
  void addToStack(const std::string &fileName, size_t lineNumber,
                  const std::string &message = "") {
    if (getIsError()) {
      payload_.asErrorValue.get()->addToStack(fileName, lineNumber, message);
    }
  }

public:
  /// Construct an Expected from an Error. The error must contain an ErrorValue.
  /// Marks the Error as checked.
  GlowExpected(GlowError error) {
    assert(error.hasErrorValue() &&
           "Must have an ErrorValue to construct an Expected from an Error");
    setErrorValue(std::move(error.takeErrorValue()), /*skipCheck*/ true);
  }

  /// Disallow construction of Expected from ErrorSuccess and ErrorEmpty.
  GlowExpected(GlowErrorSuccess) = delete;
  GlowExpected(GlowErrorEmpty) = delete;

  /// Move construct Expected<T> from a value of type OtherT as long as OtherT
  /// is convertible to T.
  template <typename OtherT>
  GlowExpected(
      OtherT &&other,
      typename std::enable_if<std::is_convertible<OtherT, T>::value>::type * =
          nullptr) {
    setValue(std::forward<OtherT>(other), /*skipCheck*/ true);
  }

  /// Move construct Expected<T> from another Expected<T>.
  GlowExpected(GlowExpected &&other) {
    if (other.getIsError()) {
      setErrorValue(std::move(other.takeErrorValue()),
                    /*skipCheck*/ true);
    } else {
      setValue(std::move(other.takeValue()), /*skipCheck*/ true);
    }
  }

  /// Move construct Expected<T> from Expected<OtherT> as long as OtherT is
  /// convertible to T.
  template <typename OtherT>
  GlowExpected(
      GlowExpected<OtherT> &&other,
      typename std::enable_if<std::is_convertible<OtherT, T>::value>::type * =
          nullptr) {
    if (other.getIsError()) {
      setErrorValue(std::move(other.takeErrorValue()),
                    /*skipCheck*/ true);
    } else {
      setValue(std::move(other.takeValue()), /*skipCheck*/ true);
    }
  }

  /// Move assign Expected<T> from another Expected<T>.
  GlowExpected &operator=(GlowExpected &&other) {
    if (other.getIsError()) {
      setErrorValue(std::move(other.takeErrorValue()));
    } else {
      setValue(std::move(other.takeValue()));
    }
    return *this;
  }

  /// Destructor for Expected, manually destroys the constents of payload_.
  ~GlowExpected() {
    if (getIsError()) {
      payload_.asErrorValue.destroy();
    } else {
      payload_.asValue.destroy();
    }
  }

  /// Overload for operator bool that returns true if no ErrorValue is
  /// contained. Marks the state as checked if no ErrorValue is contained.
  explicit operator bool() {
    bool isError = getIsError();
    if (!isError) {
      setChecked(true);
    }
    return !isError;
  }

  /// Get a reference to a value contained by payload_.
  T &get() { return *getValue(); }

  /// Get a const reference to a value contained by payload_.
  const T &get() const { return *getValue(); }

  /// Construct and \returns an Error and takes ownership of any ErrorValue in
  /// payload_. If no ErrorValue is in payload_ then return Error::success().
  /// Marks the Exected as checked no matter what.
  GlowError takeError() {
    if (getIsError()) {
      return GlowError(takeErrorValue());
    }
    setChecked(true);
    return GlowError::success();
  }

  /// \returns a pointer to the contained ErrorValue or nullptr if no
  /// ErrorValue is contained in this Expected.
  inline const GlowErrorValue *peekErrorValue() const {
    if (getIsError()) {
      return payload_.asErrorValue.get().get();
    }
    return nullptr;
  }

  T *operator->() { return getValue(); }

  const T *operator->() const { return getValue(); }

  T &operator*() { return *getValue(); }

  const T &operator*() const { return *getValue(); }

  /// NOTE: Only use for testing!
  bool isChecked_() const { return isChecked(); }
};

/// Given an Expected<T>, asserts that it contains a value T and \returns it. If
/// an ErrorValue is contained in the expected then logs this along with \p
/// fileName and \p lineNumber and aborts.
template <typename T>
T exitOnError(const char *fileName, size_t lineNumber,
              GlowExpected<T> expected) {
  if (expected) {
    return expected.takeValue();
  } else {
    auto error = expected.takeError();
    std::unique_ptr<GlowErrorValue> errorValue =
        detail::takeErrorValue(std::move(error));
    assert(errorValue != nullptr && "Expected should have a non-null "
                                    "ErrorValue if bool(expected) is false");
    errorValue->addToStack(fileName, lineNumber);
    LOG(FATAL) << "exitOnError(Expected<T>) got an unexpected ErrorValue: "
               << (*errorValue);
  }
}

/// Constructs an ErrorValue from \p args then wraps and \returns it in an
/// Error.
/// NOTE: this should not be used directly, use macros defined at the top of
/// Error.h instead.
template <typename... Args>
GlowError makeError(const char *fileName, size_t lineNumber, Args &&...args) {
  auto errorValue = std::unique_ptr<GlowErrorValue>(
      new GlowErrorValue(std::forward<Args>(args)...));
  errorValue->addToStack(fileName, lineNumber);
  return GlowError(std::move(errorValue));
}

/// Given an Error \p error, destroys the Error and returns true if an
/// ErrorValue was contained. Logs if \p log is true and uses \p fileName and \p
/// lineNumber for additional logging information. If \p warning is true then
/// the log message will replace "Error" with "Warning", this is useful for when
/// Errors are used in non-exceptional conditions.
/// NOTE: this should not be used
/// directly, use macros defined at the top of Error.h instead.
bool errorToBool(const char *fileName, size_t lineNumber, GlowError error,
                 bool log = true, bool warning = false);

/// Given an Error \p error, destroys the Error and returns a string that is the
/// result of calling log() on the ErrorValue it contained if any and "success"
/// otherwise. If \p warning is true then the log message will replace "Error"
/// with "Warning", this is useful for when Errors are used in non-exceptional
/// conditions.
/// NOTE: this should not be used directly, use macros defined at
/// the top of Error.h instead.
std::string errorToString(const char *fileName, size_t lineNumber,
                          GlowError error, bool warning = false);

/// Given an Error \p error, destroys the Error. Logs if \p log is true and uses
/// \p fileName and \p lineNumber for additional logging information. If \p
/// warning is true then the log message will replace "Error" with "Warning",
/// this is useful for when Errors are used in non-exceptional conditions.
/// NOTE: this should not be used directly, use macros defined at the top of
/// Error.h instead.
void errorToVoid(const char *fileName, size_t lineNumber, GlowError error,
                 bool log = true, bool warning = false);
} // namespace detail

/// This class holds an Error provided via the add method. If an Error is
/// added when the class already holds an Error, it will discard the new Error
/// in favor of the original one. All methods in OneErrOnly are thread-safe.
class OneErrOnly {
  Error err_ = Error::empty();
  std::mutex m_;

public:
  /// Add a new Error \p err to be stored. If an existing Error has
  /// already been added then the contents of the new error will be logged and
  /// the new err will be discarded. \returns true if \p err was stored and
  /// \returns false otherwise. If \p err is an empty Error then does nothing
  /// and \returns false;
  bool set(Error err);

  /// \returns the stored Error clearing out the storage of the class.
  Error get();

  /// \returns true if contains an Error and false otherwise.
  bool containsErr();
};

} // end namespace glow

#endif // GLOW_SUPPORT_ERROR_H
