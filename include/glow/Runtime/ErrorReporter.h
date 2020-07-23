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
#ifndef GLOW_RUNTIME_ERROR_REPORTER_H
#define GLOW_RUNTIME_ERROR_REPORTER_H

#include <memory>
#include <string>
#include <vector>

namespace glow {

/// Interface for exporting runtime statistics.  The base implementation
/// delegates to any subclass registered via `ErrorReporter`.
class ErrorReporter {
public:
  /// Dtor.
  virtual ~ErrorReporter() = default;

  /// Report error to some sink.
  virtual void report(const std::string &msg) = 0;
};

/// Registry of ErrorReporters..
class ErrorReporterRegistry final {
public:
  /// Start all the error reporters.
  void report(const std::string &msg) {
    for (auto *r : reporters_) {
      r->report(msg);
    }
  }

  /// Register a ErrorReporter.
  void registerErrorReporter(ErrorReporter *reporter);

  /// Revoke a ErrorReporter.
  void revokeErrorReporter(ErrorReporter *reporter);

  /// Static singleton ErrorReporterRegistry.
  static std::shared_ptr<ErrorReporterRegistry> ErrorReporters();

private:
  /// Registered ErrorReporter..
  std::vector<ErrorReporter *> reporters_;
};

} // namespace glow

#endif // GLOW_RUNTIME_ERROR_REPORTER_H
