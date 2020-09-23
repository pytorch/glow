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

#include "gtest/gtest.h"

using namespace glow;

TEST(Error, BasicError) {
  auto err = MAKE_ERR("some error");
  EXPECT_TRUE(ERR_TO_BOOL(std::move(err)));
}

TEST(Error, ErrorSuccess) {
  auto err = Error::success();
  EXPECT_FALSE(ERR_TO_BOOL(std::move(err)));
}

TEST(Error, ErrorSuccessReturn) {
  auto f = []() -> Error { return Error::success(); };
  auto err = f();
  EXPECT_FALSE(ERR_TO_BOOL(std::move(err)));
}

TEST(Error, ErrorString) {
  const char *msg = "some error";
  auto err = MAKE_ERR(msg);
  auto str = ERR_TO_STRING(std::move(err));
  EXPECT_NE(str.find(msg), std::string::npos)
      << "Error should preserve the given message";
}

TEST(Error, BasicOpaque) {
  using glow::detail::Opaque;

  Opaque<int64_t> opaqueInt;
  opaqueInt.set(42);
  EXPECT_EQ(opaqueInt.get(), 42);
}

TEST(Error, OpaqueDestructorCall) {
  using glow::detail::Opaque;

  /// Struct that takes a pointer to a boolean in it's constructor and set's the
  /// boolean to true when it is destructed.
  struct SetFlagOnDestruct {
    bool *b_ = nullptr;

    SetFlagOnDestruct(bool *b) : b_(b) {}

    ~SetFlagOnDestruct() {
      if (b_ != nullptr) {
        *b_ = true;
      }
    }

    SetFlagOnDestruct(SetFlagOnDestruct &&other) { std::swap(b_, other.b_); }

    SetFlagOnDestruct &operator=(SetFlagOnDestruct &&other) {
      std::swap(b_, other.b_);
      return *this;
    }
  };

  bool b1 = false;
  SetFlagOnDestruct flagSetter1(&b1);

  bool b2 = false;
  SetFlagOnDestruct flagSetter2(&b2);

  {
    Opaque<SetFlagOnDestruct> opaque1;
    opaque1.set(std::move(flagSetter1));
  }

  {
    Opaque<SetFlagOnDestruct> opaque2;
    opaque2.set(std::move(flagSetter2));
    opaque2.destroy();
  }

  ASSERT_FALSE(b1) << "The destructor of the contents of Opaque shouldn't be "
                      "called when Opaque is destroyed";

  EXPECT_TRUE(b2) << "The destructor of the contents of Opaque should be "
                     "called when Opaque.destroy() is called";

  // Check size and alignment of Opaque<SetFlagOnDestruct>
  EXPECT_EQ(sizeof(Opaque<SetFlagOnDestruct>), sizeof(SetFlagOnDestruct));
  EXPECT_EQ(alignof(Opaque<SetFlagOnDestruct>), alignof(SetFlagOnDestruct));
}

TEST(Error, ExpectedValue) {
  Expected<std::string> stringOrErr("hello world");
  if (stringOrErr) {
    EXPECT_EQ(stringOrErr.get(), "hello world");
  } else {
    FAIL() << "This expected should have a value";
  }
}

TEST(Error, ExpectedError) {
  const char *msg = "some error";
  auto err = MAKE_ERR(msg);
  Expected<int> intOrErr = std::move(err);
  if (intOrErr) {
    FAIL() << "This expected should not have a value";
  } else {
    auto err2 = intOrErr.takeError();
    auto str = ERR_TO_STRING(std::move(err2));
    EXPECT_NE(str.find(msg), std::string::npos)
        << "Expected should preserve the given message";
  }
}

TEST(Error, ExpectedTakeErrorWithoutError) {
  Expected<int> intOrErr(42);
  auto err = intOrErr.takeError();
  EXPECT_FALSE(err);
}

TEST(Error, EmptyErrors) {
  Error err = Error::empty();

  auto f = [&]() { err = MAKE_ERR("something"); };

  f();

  EXPECT_TRUE(ERR_TO_BOOL(std::move(err)));
}

// Creating an unused OneErrOnly should be safe.
TEST(Error, UntouchedOneErrOnly) { OneErrOnly foo; }

TEST(Error, ExpectedConversion) {
  auto foo = []() -> Expected<int32_t> { return 42; };

  auto bar = [&]() -> Expected<int64_t> { return foo(); };

  int64_t barRes;
  ASSIGN_VALUE_OR_FAIL_TEST(barRes, bar());

  EXPECT_EQ(barRes, 42);
}

TEST(Error, ReturnIfExpectedIsErr) {
  auto retErr = [&]() -> Expected<int> { return MAKE_ERR("Error!"); };
  auto test = [&]() -> Error {
    RETURN_IF_EXPECTED_IS_ERR(retErr());
    return Error::success();
  };

  EXPECT_TRUE(ERR_TO_BOOL(test()));
}

TEST(Error, PeekError) {
  const char *msg = "some error";
  auto err = MAKE_ERR(msg);
  auto str = err.peekErrorValue()->logToString();
  EXPECT_NE(str.find(msg), std::string::npos)
      << "Error should preserve the given message";
#ifndef NDEBUG
  EXPECT_FALSE(err.isChecked_());
#endif
  ERR_TO_VOID(std::move(err));
}

TEST(Error, PeekExpected) {
  const char *msg = "some error";
  Expected<int> intOrErr = MAKE_ERR(msg);
  auto str = intOrErr.peekErrorValue()->logToString();
  EXPECT_NE(str.find(msg), std::string::npos)
      << "Error should preserve the given message";
#ifndef NDEBUG
  EXPECT_FALSE(intOrErr.isChecked_());
#endif
  ERR_TO_VOID(intOrErr.takeError());
}

TEST(Error, WarningString) {
  const char *msg = "some message";
  auto err = MAKE_ERR(msg);
  auto str = ERR_TO_STRING(std::move(err), /*warning*/ true);
  EXPECT_NE(str.find("Warning"), std::string::npos)
      << "Expect warning to be present in message";
}
