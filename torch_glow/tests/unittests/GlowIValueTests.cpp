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

#include "GlowIValue.h"
#include <ATen/ATen.h>

#include "glow/Runtime/HostManager/HostManager.h"

#include <gtest/gtest.h>

using namespace glow;

TEST(GlowIValueTests, NoneTest) {
  GlowIValue ival;
  ival.fromNone();
  EXPECT_TRUE(ival.isNone());
}

TEST(GlowIValueTests, TensorTest) {
  glow::Tensor t = {1.0, 2.0, 3.0};
  glow::Tensor refT = t.clone();

  GlowIValue ival;
  ival.fromTensor(std::move(t));

  ASSERT_TRUE(ival.isTensor());

  glow::Tensor *tPtr;
  ASSIGN_VALUE_OR_FAIL_TEST(tPtr, ival.toTensor());

  EXPECT_TRUE(tPtr->isEqual(refT));
}

TEST(GlowIValueTests, DoubleTest) {
  GlowIValue ival;
  ival.fromDouble(3.14);

  ASSERT_TRUE(ival.isDouble());

  double res;
  ASSIGN_VALUE_OR_FAIL_TEST(res, ival.toDouble());
  EXPECT_EQ(res, 3.14);
}

TEST(GlowIValueTests, IntTest) {
  GlowIValue ival;
  ival.fromInt(96);

  ASSERT_TRUE(ival.isInt());

  int64_t res;
  ASSIGN_VALUE_OR_FAIL_TEST(res, ival.toInt());
  EXPECT_EQ(res, 96);
}

TEST(GlowIValueTests, BoolTest) {
  GlowIValue ival;
  ival.fromBool(true);

  ASSERT_TRUE(ival.isBool());

  bool res;
  ASSIGN_VALUE_OR_FAIL_TEST(res, ival.toBool());
  EXPECT_EQ(res, true);
}

TEST(GlowIValueTests, IntListTest) {
  std::vector<int64_t> l = {42, 44, 46};
  std::vector<int64_t> lRef = l;

  GlowIValue ival;
  ival.fromIntList(std::move(l));

  ASSERT_TRUE(ival.isIntList());

  std::vector<int64_t> *res;
  ASSIGN_VALUE_OR_FAIL_TEST(res, ival.toIntList());
  EXPECT_EQ(*res, lRef);
}

TEST(GlowIValueTests, DoubleListTest) {
  std::vector<double> l = {3.0, 3.5, 4.0};
  std::vector<double> lRef = l;

  GlowIValue ival;
  ival.fromDoubleList(std::move(l));

  ASSERT_TRUE(ival.isDoubleList());

  std::vector<double> *res;
  ASSIGN_VALUE_OR_FAIL_TEST(res, ival.toDoubleList());
  EXPECT_EQ(*res, lRef);
}

TEST(GlowIValueTests, BoolListTest) {
  std::vector<bool> l = {true, false, true};
  std::vector<bool> lRef = l;

  GlowIValue ival;
  ival.fromBoolList(std::move(l));

  ASSERT_TRUE(ival.isBoolList());

  std::vector<bool> *res;
  ASSIGN_VALUE_OR_FAIL_TEST(res, ival.toBoolList());
  EXPECT_EQ(*res, lRef);
}

TEST(GlowIValueTests, TupleTest) {
  std::vector<GlowIValue> iVals(2);

  iVals[0].fromInt(33);
  iVals[1].fromInt(44);

  GlowIValue ival;
  ival.fromTuple(std::move(iVals));

  ASSERT_TRUE(ival.isTuple());

  std::vector<GlowIValue> *res;
  ASSIGN_VALUE_OR_FAIL_TEST(res, ival.toTuple());

  ASSERT_TRUE(res->at(0).isInt());
  ASSERT_TRUE(res->at(1).isInt());

  int64_t intRes;
  ASSIGN_VALUE_OR_FAIL_TEST(intRes, res->at(0).toInt());
  EXPECT_EQ(intRes, 33);

  ASSIGN_VALUE_OR_FAIL_TEST(intRes, res->at(1).toInt());
  EXPECT_EQ(intRes, 44);
}

TEST(GlowIValueTests, StringTest) {
  GlowIValue ival;
  ival.fromString("hello world");

  ASSERT_TRUE(ival.isString());

  std::string *res;
  ASSIGN_VALUE_OR_FAIL_TEST(res, ival.toString());
  EXPECT_EQ(*res, "hello world");
}

TEST(GLOWIValueTests, NodeValueListTest) {
  GlowIValue ival;
  std::unique_ptr<Module> module = glow::make_unique<Module>();
  glow::Tensor t1 = {1, 2, 3};
  glow::Tensor t2 = {4, 5, 6, 7, 8, 9};
  auto n1 = module->createConstant("c1", std::move(t1))->getOutput();
  auto n2 = module->createConstant("c2", std::move(t2))->getOutput();

  std::vector<glow::NodeValue> v;
  v.push_back(n1);
  v.push_back(n2);

  ival.fromNodeValueList(v);
  ASSERT_TRUE(ival.isNodeValueList());

  std::vector<glow::NodeValue> *vRes;
  ASSIGN_VALUE_OR_FAIL_TEST(vRes, ival.toNodeValueList());

  EXPECT_EQ(vRes->size(), 2);
  EXPECT_EQ((*vRes)[0].getNode()->getKind(), Kinded::Kind::ConstantKind);
  EXPECT_EQ((*vRes)[1].getNode()->getKind(), Kinded::Kind::ConstantKind);

  glow::Constant *c1 = llvm::dyn_cast<glow::Constant>((*vRes)[0].getNode());
  glow::Constant *c2 = llvm::dyn_cast<glow::Constant>((*vRes)[1].getNode());
  const auto h1 = c1->getPayload().getHandle<float>();
  const auto h2 = c2->getPayload().getHandle<float>();

  for (int i = 1; i <= 3; i++) {
    EXPECT_EQ(h1.raw(i - 1), i);
  }
  for (int i = 4; i <= 9; i++) {
    EXPECT_EQ(h2.raw(i - 4), i);
  }
}

TEST(GlowIValueTests, PTTensorTest) {
  auto t = at::empty({3}, at::kCPU);
  t[0] = 2.0;
  t[1] = 4.0;
  t[2] = 6.0;

  auto tRef = t.clone();

  GlowIValue ival;
  ival.fromPTTensor(std::move(t));

  ASSERT_TRUE(ival.isPTTensor());

  at::Tensor *tPtr;
  ASSIGN_VALUE_OR_FAIL_TEST(tPtr, ival.toPTTensor());

  EXPECT_TRUE(tPtr->equal(tRef));
}

TEST(GlowIValueTests, GenericMapTest) {
  GlowIValueMap m;

  GlowIValue ival, k1, k2, v1, v2;

  k1.fromString("foo");
  k2.fromString("bar");

  v1.fromInt(42);
  v2.fromInt(12321);

  m.emplace(std::move(k1), std::move(v1));
  m.emplace(std::move(k2), std::move(v2));

  ival.fromGenericMap(std::move(m));

  ASSERT_TRUE(ival.isGenericMap());

  GlowIValueMap *mPtr;
  ASSIGN_VALUE_OR_FAIL_TEST(mPtr, ival.toGenericMap());

  GlowIValue k;
  k.fromString("bar");

  auto vIt = mPtr->find(k);
  ASSERT_TRUE(vIt != mPtr->end());

  int64_t v;
  ASSIGN_VALUE_OR_FAIL_TEST(v, vIt->second.toInt());
  EXPECT_EQ(v, 12321);
}

TEST(GlowIValueTests, GenericMapFromIValueTest) {
  c10::Dict<std::string, int64_t> m;
  m.insert("foo", 42);
  m.insert("bar", 12321);

  c10::IValue ival(std::move(m));

  GlowIValue glowIVal;
  FAIL_TEST_IF_ERR(glowIVal.fromIValue(ival));

  ASSERT_TRUE(glowIVal.isGenericMap());

  GlowIValueMap *mPtr;
  ASSIGN_VALUE_OR_FAIL_TEST(mPtr, glowIVal.toGenericMap());

  GlowIValue k;
  k.fromString("bar");

  auto vIt = mPtr->find(k);
  ASSERT_TRUE(vIt != mPtr->end());

  int64_t v;
  ASSIGN_VALUE_OR_FAIL_TEST(v, vIt->second.toInt());
  EXPECT_EQ(v, 12321);
}

TEST(GlowIValueTests, SwitchTagTest) {
  GlowIValue ival;
  ival.fromInt(42);

  ASSERT_TRUE(ival.isInt());

  int64_t intRes;
  ASSIGN_VALUE_OR_FAIL_TEST(intRes, ival.toInt());
  EXPECT_EQ(intRes, 42);

  ival.fromDouble(11.5);

  ASSERT_TRUE(ival.isDouble());

  double doubleRes;
  ASSIGN_VALUE_OR_FAIL_TEST(doubleRes, ival.toDouble());
  EXPECT_EQ(doubleRes, 11.5);
}
