#include "glow/Interpreter/Interpreter.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"

#include "gtest/gtest.h"

#include <cassert>
#include <string>

using namespace glow;

TEST(Interpreter, init) {

  Interpreter IP;

  auto &builder = IP.getBuilder();
  auto *v = builder.createStaticVariable(glow::ElemKind::FloatTy, {320, 200});
  auto *t = new Tensor(glow::ElemKind::FloatTy, {320, 200});
  IP.registerTensor(v, t);

  /// Check that value-tensor registery works.
  EXPECT_EQ(IP.getTensorForValue(v), t);
}
