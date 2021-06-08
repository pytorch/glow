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

#include "glow/Runtime/ErrorReporter.h"

#include <algorithm>

namespace glow {

void ErrorReporterRegistry::registerErrorReporter(ErrorReporter *r) {
  reporters_.push_back(r);
}

void ErrorReporterRegistry::revokeErrorReporter(ErrorReporter *r) {
  reporters_.erase(std::remove(reporters_.begin(), reporters_.end(), r),
                   reporters_.end());
}

std::shared_ptr<ErrorReporterRegistry> ErrorReporterRegistry::ErrorReporters() {
  static auto reporters = std::make_shared<ErrorReporterRegistry>();
  return reporters;
}

detail::GlowError reportOnError(detail::GlowError error) {
  if (error) {
    auto errorValue = error.peekErrorValue();
    assert(errorValue != nullptr &&
           "Error should have a non-null ErrorValue if bool(error) is true");

    auto reporters = ErrorReporterRegistry::ErrorReporters();
    if (reporters) {
      reporters->report(errorValue->logToString());
    }
  }

  return error;
}

} // namespace glow
