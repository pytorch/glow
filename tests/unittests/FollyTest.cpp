#include <gtest/gtest.h>
// Format.h needs to be after gtest.h It brings in a folly Windows.h header
// which then breaks the build on VS.
#include <folly/Format.h>
#include <sstream>

TEST(Folly, Format) {
  std::ostringstream s;
  s << folly::format("The answer is {}", 42);
  ASSERT_TRUE(s.str() == "The answer is 42");
}
