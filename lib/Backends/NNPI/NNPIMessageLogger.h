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

#ifndef GLOW_BACKENDS_NNPI_NNPIMESSAGELOGGER_H
#define GLOW_BACKENDS_NNPI_NNPIMESSAGELOGGER_H

#include "nnpi_transformer.h"

namespace glow {
/// NNPI logging stream controller.
class NNPIMessageLogger {
public:
  static NNPIMessageLogger &getInstance() {
    static NNPIMessageLogger instance;
    return instance;
  }

  void setLogLevel(NNPI_LOG_LEVEL level);

private:
  NNPIMessageLogger();
  static uint64_t messagesWriteHandler(const void *ptr, uint64_t size,
                                       uint64_t count, void *userData);
  NNPIStream messagesStream_;
};

} // end namespace glow
#endif // GLOW_BACKENDS_NNPI_NNPIMESSAGELOGGER_H
