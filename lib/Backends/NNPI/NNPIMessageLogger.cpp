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

#include "NNPIMessageLogger.h"
#include "Importer.h"
#include <cstring>
#include <glog/logging.h>

using namespace glow;

void NNPIMessageLogger::setLogLevel(NNPI_LOG_LEVEL level) {
  NNPISetLogLevel(level);
}

NNPIMessageLogger::NNPIMessageLogger() {
  std::memset(&messagesStream_, 0, sizeof(NNPIStream));
  messagesStream_.writeCallback = NNPIMessageLogger::messagesWriteHandler;

#ifdef NDEBUG
  NNPI_LOG_LEVEL logLevel = NNPI_LOG_LEVEL_ERROR;
#else  // NDEBUG
  NNPI_LOG_LEVEL logLevel = NNPI_LOG_LEVEL_DEBUG;
#endif // NDEBUG
  logLevel = NNPIEnvVariables::getVarLogLevel("NNPI_LOG_LEVEL", logLevel);
  NNPISetLogLevel(logLevel);
  NNPISetLogStream(&messagesStream_);
}

uint64_t NNPIMessageLogger::messagesWriteHandler(const void *ptr, uint64_t size,
                                                 uint64_t /*count*/,
                                                 void * /*userData*/) {
  LOG(INFO) << "[NNPI_LOG]" << reinterpret_cast<const char *>(ptr);
  return size;
}
