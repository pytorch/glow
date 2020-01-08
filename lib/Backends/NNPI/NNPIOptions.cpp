/*
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

#include "NNPIOptions.h"
#include "DebugMacros.h"
#include "nnpi_transformer.h"
#include "nnpi_transformer_types.h"
#include <cstdio>
#include <glog/logging.h>
#include <sstream>

using namespace glow;

static bool isUInt(std::string var, size_t startOffset = 0) {
  if (!var.empty()) {
    bool foundNoDigit =
        (std::find_if(var.begin() + startOffset, var.end(),
                      [](char c) { return !std::isdigit(c); }) != var.end());
    return !foundNoDigit;
  }
  return false;
}

static bool isInt(std::string var) {
  if (!var.empty()) {
    size_t startOffset = 0;
    if (var[0] == '-') {
      startOffset = 1;
    }
    return isUInt(var, startOffset);
  }
  return false;
}

static uint64_t messagesWriteHandler(const void *ptr, uint64_t size,
                                     uint64_t /*count*/, void * /*userData*/) {
  LOG(INFO) << "[NNPI_LOG]" << reinterpret_cast<const char *>(ptr);
  return size;
}

std::string NNPIOptions::getFromEnv(std::string envName, std::string defVal) {
  std::string var;
  char *pEnvVar = getenv(envName.c_str());
  return (pEnvVar ? std::string(pEnvVar) : defVal);
}

template <>
std::string NNPIOptions::getStringAsType<std::string>(std::string sVal) {
  return sVal;
}

template <> bool NNPIOptions::getStringAsType<bool>(std::string sVal) {
  return sVal == "1";
}

template <> int NNPIOptions::getStringAsType<int>(std::string sVal) {
  if (isInt(sVal)) {
    return std::stoi(sVal);
  }
  return -1;
}

template <> unsigned NNPIOptions::getStringAsType<unsigned>(std::string sVal) {
  if (isInt(sVal)) {
    return (unsigned)std::stol(sVal);
  }
  return 0;
}

std::string NNPIOptions::dumpStatus() {
  std::stringstream desc;
  desc << "\nNNPI " << getOptionsName().data() << " variables\n";
  for (auto key : loadedOptions_.keys()) {
    desc << " " << key.str() << "=" << loadedOptions_[key.str()] << "\n";
  }
  return desc.str();
}

llvm::StringMap<std::string> NNPIOptions::getSupportedOptions() {
  return supportedOptions_;
}

void NNPICompilationOptions::setLogLevel(int logLevel) {
  // We have only one log level for NNPI compilation. Setting it will change
  // the level for all compilation instances.
  NNPI_LOG_LEVEL level;
  switch (logLevel) {
  case 0:
    level = NNPI_LOG_LEVEL_VERBOSE;
    break;
  case 1:
    level = NNPI_LOG_LEVEL_DEBUG;
    break;
  case 2:
    level = NNPI_LOG_LEVEL_ASSERT;
    break;
  case 3:
    level = NNPI_LOG_LEVEL_INFO;
    break;
  case 4:
    level = NNPI_LOG_LEVEL_WARNING;
    break;
  case 5:
    level = NNPI_LOG_LEVEL_CRITICAL;
    break;
  case 6:
    level = NNPI_LOG_LEVEL_ERROR;
    break;
  default:
    level = NNPI_LOG_LEVEL_USER;
  }
  nnpiSetLogLevel(level);
  static std::mutex logSyncMutex;
  std::lock_guard<std::mutex> lk(logSyncMutex);
  static bool logStreamSet = false;
  if (logStreamSet) {
    // Set log stream only once.
    return;
  }
  static NNPIStream logStream;
  std::memset(&logStream, 0, sizeof(NNPIStream));
  logStream.writeCallback = messagesWriteHandler;
  nnpiSetLogStream(&logStream);
  logStreamSet = true;
}
