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

#include "glow/Support/Error.h"

#include <sstream>

namespace glow {
bool OneErrOnly::set(Error err) {
  // Don't do anything in the case of empty Error.
  if (!err) {
    return false;
  }

  std::unique_lock<std::mutex> lock(m_);

  if (!err_) {
    err_ = std::move(err);
    return true;
  } else {
    // No update happening so don't need the lock any more.
    lock.unlock();
    LOG(ERROR) << "OneErrOnly already has an Error, discarding new Error: "
               << ERR_TO_STRING(std::move(err));
    return false;
  }
}

Error OneErrOnly::get() {
  std::unique_lock<std::mutex> lock(m_);
  auto err = std::move(err_);
  return err;
}

bool OneErrOnly::containsErr() {
  std::unique_lock<std::mutex> lock(m_);
  return static_cast<bool>(err_);
}

namespace detail {
std::string GlowErrorValue::logToString(bool warning) const {
  std::stringstream ss;
  log(ss, warning);
  return ss.str();
}

std::string GlowErrorValue::errorCodeToString(const ErrorCode &ec) {
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
  case ErrorCode::PARTITIONER_ERROR:
    return "PARTITIONER_ERROR";
  case ErrorCode::RUNTIME_ERROR:
    return "RUNTIME_ERROR";
  case ErrorCode::RUNTIME_DEFERRED_WEIGHT_ERROR:
    return "RUNTIME_DEFERRED_WEIGHT_ERROR";
  case ErrorCode::RUNTIME_OUT_OF_DEVICE_MEMORY:
    return "RUNTIME_OUT_OF_DEVICE_MEMORY";
  case ErrorCode::RUNTIME_NET_NOT_FOUND:
    return "RUNTIME_NET_NOT_FOUND";
  case ErrorCode::RUNTIME_REQUEST_REFUSED:
    return "RUNTIME_REQUEST_REFUSED";
  case ErrorCode::RUNTIME_DEVICE_NOT_FOUND:
    return "RUNTIME_DEVICE_NOT_FOUND";
  case ErrorCode::RUNTIME_DEVICE_NONRECOVERABLE:
    return "RUNTIME_DEVICE_NONRECOVERABLE";
  case ErrorCode::RUNTIME_NET_BUSY:
    return "RUNTIME_NET_BUSY";
  case ErrorCode::DEVICE_FEATURE_NOT_SUPPORTED:
    return "DEVICE_FEATURE_NOT_SUPPORTED";
  case ErrorCode::COMPILE_UNSUPPORTED_NODE_AFTER_OPTIMIZE:
    return "COMPILE_UNSUPPORTED_NODE_AFTER_OPTIMIZE";
  case ErrorCode::COMPILE_CONTEXT_MALFORMED:
    return "COMPILE_CONTEXT_MALFORMED";
  case ErrorCode::MODEL_WRITER_INVALID_FILENAME:
    return "MODEL_WRITER_INVALID_FILENAME";
  case ErrorCode::MODEL_WRITER_SERIALIZATION_ERROR:
    return "MODEL_WRITER_SERIALIZATION_ERROR";
  case ErrorCode::COMPILE_UNSUPPORTED_IR_AFTER_GENERATE:
    return "COMPILE_UNSUPPORTED_IR_AFTER_GENERATE";
  case ErrorCode::COMPILE_UNSUPPORTED_IR_AFTER_OPTIMIZE:
    return "COMPILE_UNSUPPORTED_IR_AFTER_OPTIMIZE";
  };
  LOG(FATAL) << "Unsupported ErrorCode";
}

std::unique_ptr<GlowErrorValue> takeErrorValue(GlowError error) {
  return error.takeErrorValue();
}

void exitOnError(const char *fileName, size_t lineNumber, GlowError error) {
  if (error) {
    std::unique_ptr<GlowErrorValue> errorValue =
        detail::takeErrorValue(std::move(error));
    assert(errorValue != nullptr &&
           "Error should have a non-null ErrorValue if bool(error) is true");
    errorValue->addToStack(fileName, lineNumber);
    LOG(FATAL) << "exitOnError(Error) got an unexpected ErrorValue: "
               << (*errorValue);
  }
}

bool errorToBool(const char *fileName, size_t lineNumber, GlowError error,
                 bool log, bool warning) {
  std::unique_ptr<GlowErrorValue> errorValue =
      detail::takeErrorValue(std::move(error));
  if (errorValue) {
    if (log) {
      errorValue->addToStack(fileName, lineNumber);
      LOG(ERROR) << "Converting Error to bool: "
                 << errorValue->logToString(warning);
    }
    return true;
  } else {
    return false;
  }
}

std::string errorToString(const char *fileName, size_t lineNumber,
                          GlowError error, bool warning) {
  std::unique_ptr<GlowErrorValue> errorValue =
      detail::takeErrorValue(std::move(error));
  if (errorValue) {
    errorValue->addToStack(fileName, lineNumber);
    return errorValue->logToString(warning);
  } else {
    return "success";
  }
}

void errorToVoid(const char *fileName, size_t lineNumber, GlowError error,
                 bool log, bool warning) {
  errorToBool(fileName, lineNumber, std::move(error), log, warning);
}

GlowError::GlowError(GlowErrorEmpty &&other) {
  setErrorValue(std::move(other.errorValue_), /*skipCheck*/ true);
  setChecked(true);
  other.setChecked(true);
}

} // namespace detail
} // namespace glow
