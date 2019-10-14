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

#ifndef GLOW_TESTING_STRCHECK_H
#define GLOW_TESTING_STRCHECK_H

#include "llvm/ADT/StringRef.h"

#include <string>
#include <vector>

// Forward decls from googletest.
namespace testing {
class AssertionResult;
}

namespace glow {

/// Helper class for finding ordered substrings in a larger string.
///
/// The StrCheck class is intended to be used in googletest unit tests:
///
///   EXPECT_TRUE(StrCheck(asm).check("foo").no("baz").sameln("arg=7"));
///
/// A typical use is to search for expected instruction patterns in
/// disassembled code generated for a test case.
class StrCheck {
public:
  StrCheck(llvm::StringRef input) : input_(input) {}

  /// Check that `needle` appears in the input at a point after the previous
  /// match.
  StrCheck &check(llvm::StringRef needle);

  /// Check that `needle` appears on the same line as the previous match. (And
  /// following the previous match).
  StrCheck &sameln(llvm::StringRef needle);

  /// Check that `needle` appears on the next line after the previous match.
  StrCheck &nextln(llvm::StringRef needle);

  /// Check that no instances of `needle` appear before the next positive match.
  StrCheck &no(llvm::StringRef needle);

private:
  const std::string input_;
  size_t pos_ = 0;
  std::string errors_;
  std::vector<std::string> nos_;

  size_t findEol(size_t p) const;
  size_t findEol() const;
  StrCheck &match(size_t pos, size_t size);
  StrCheck &fail(const char *msg, llvm::StringRef needle);

public:
  // Allow implicit conversion to allow a StrCheck to be used in EXPECT_TRUE().
  operator testing::AssertionResult();
  operator bool();
};

} // namespace glow

#endif // GLOW_TESTING_STRCHECK_H
