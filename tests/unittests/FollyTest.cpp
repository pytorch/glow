#include <folly/Format.h>
#include <gtest/gtest.h>
#include <sstream>

TEST(Folly, Format) {
  std::ostringstream s;
  s << folly::format("The answer is {}", 42);
  ASSERT_TRUE(s.str() == "The answer is 42");
}

