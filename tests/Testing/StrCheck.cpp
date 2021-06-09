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

#include "glow/Testing/StrCheck.h"
#include "gtest/gtest.h"

using glow::StrCheck;

StrCheck &StrCheck::check(llvm::StringRef needle) {
  size_t found = input_.find(needle.str(), pos_);
  return found != std::string::npos ? match(found, needle.size())
                                    : fail("not found: check", needle);
}

StrCheck &StrCheck::sameln(llvm::StringRef needle) {
  size_t found = input_.find(needle.str(), pos_);
  return found < findEol() ? match(found, needle.size())
                           : fail("not found: sameln", needle);
}

StrCheck &StrCheck::nextln(llvm::StringRef needle) {
  size_t eol = findEol();
  size_t found = input_.find(needle.str(), eol);
  return found < findEol(eol) ? match(found, needle.size())
                              : fail("not found: nextln", needle);
}

StrCheck &StrCheck::no(llvm::StringRef needle) {
  nos_.push_back(needle.str());
  return *this;
}

StrCheck &StrCheck::match(size_t at, size_t size) {
  // We have a positive match. Check if any of the no()'s appeared between
  // `pos_` and `at`.
  for (const auto &no : nos_) {
    if (pos_ + no.size() > at)
      continue;
    if (input_.find(no, pos_) <= at - no.size())
      fail("matched not", no);
  }
  nos_.clear();

  pos_ = at + size;
  return *this;
}

StrCheck &StrCheck::fail(const char *msg, llvm::StringRef needle) {
  errors_ += ">>> ";
  errors_ += msg;
  errors_ += "(\"";
  errors_ += needle;
  errors_ += "\")\n";
  return *this;
}

size_t StrCheck::findEol(size_t p) const {
  size_t eol = input_.find('\n', p);
  return eol == std::string::npos ? input_.size() : eol + 1;
}

size_t StrCheck::findEol() const { return findEol(pos_); }

StrCheck::operator testing::AssertionResult() {
  if (*this)
    return testing::AssertionSuccess();
  else
    return testing::AssertionFailure() << "Failed StrCheck in:\n"
                                       << input_ << '\n'
                                       << errors_;
}

StrCheck::operator bool() {
  // Match up to the end of the input to make sure we catch any trailing no()'s.
  match(input_.size(), 0);
  return errors_.empty();
}
