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

// This file tests the basic functionality of the float16 type.
// This is by no mean a test to show the IEEE 754 compliance!

#include "glow/Base/TaggedList.h"
#include "gtest/gtest.h"

#include <iterator>

using namespace glow;

namespace {
class Node : public TaggedListNode<Node> {
public:
  Node(int id) : id(id) {}
  int id;
  int adds = 0;
};
} // namespace

struct ListTraits : TaggedListTraits<Node> {
  void addNodeToList(Node *node) { node->adds++; }
  void removeNodeFromList(Node *node) { node->adds--; }
};

using List = TaggedList<Node, ListTraits>;

TEST(TaggedList, empty) {
  List ls;
  EXPECT_TRUE(ls.empty());
  EXPECT_EQ(ls.size(), 0);
  EXPECT_TRUE(ls.begin() == ls.end());
  EXPECT_FALSE(ls.begin() != ls.end());
}

TEST(TaggedList, empty_const) {
  const List ls;
  EXPECT_TRUE(ls.empty());
  EXPECT_EQ(ls.size(), 0);
  EXPECT_TRUE(ls.begin() == ls.end());
  EXPECT_FALSE(ls.begin() != ls.end());
}

TEST(TaggedList, insert) {
  List ls;
  Node n7(7);
  auto ins = ls.insert(ls.end(), &n7);

  EXPECT_EQ(ls.size(), 1);
  EXPECT_FALSE(ls.empty());
  EXPECT_TRUE(ls.begin() == ins);
  EXPECT_FALSE(ls.begin() != ins);
  EXPECT_TRUE(ins != ls.end());
  EXPECT_FALSE(ins == ls.end());
  EXPECT_TRUE(&*ins == &n7);
  EXPECT_EQ(ins->id, 7);

  // Trait callback.
  EXPECT_EQ(n7.adds, 1);

  // Insert before a real elem: [n8 n7]
  Node n8(8);
  auto in2 = ls.insert(ins, &n8);
  EXPECT_EQ(ls.size(), 2);
  EXPECT_TRUE(&*in2 == &n8);
  EXPECT_TRUE(ls.begin() == in2);
  EXPECT_TRUE(ls.begin() != ins);
  EXPECT_EQ(ins->id, 7);
  EXPECT_EQ(in2->id, 8);

  // Insert in the middle: [n8 n9 n7]
  Node n9(9);
  auto in3 = ls.insert(ins, &n9);
  EXPECT_TRUE(ls.begin() != in3);
  EXPECT_TRUE(ls.end() != in3);
  EXPECT_EQ(in3->id, 9);

  // Iterate over the list in both directions.
  auto i = ls.begin();
  EXPECT_EQ(i->id, 8);
  ++i;
  EXPECT_EQ(i++->id, 9);
  EXPECT_EQ(i->id, 7);
  ++i;
  EXPECT_TRUE(i == ls.end());
  EXPECT_EQ((--i)->id, 7);
  --i;
  EXPECT_EQ((i--)->id, 9);
  EXPECT_EQ(i->id, 8);
  EXPECT_TRUE(i == ls.begin());

  // Same, with const_iterator.
  {
    const auto &cls = ls;
    auto i = cls.begin();
    EXPECT_EQ(i->id, 8);
    ++i;
    EXPECT_EQ(i++->id, 9);
    EXPECT_EQ(i->id, 7);
    ++i;
    EXPECT_TRUE(i == cls.end());
    EXPECT_EQ((--i)->id, 7);
    --i;
    EXPECT_EQ((i--)->id, 9);
    EXPECT_EQ(i->id, 8);
    EXPECT_TRUE(i == cls.begin());
  }

  // Destructor crashes trying to delete stack nodes.
  ls.clearAndLeakNodesUnsafely();
}

TEST(TaggedList, remove) {
  List ls;
  Node n1(1), n2(2), n3(3);
  auto i1 = ls.insert(ls.end(), &n1);
  ls.insert(ls.end(), &n2);
  auto i3 = ls.insert(ls.end(), &n3);
  EXPECT_EQ(ls.size(), 3);
  EXPECT_EQ(n1.adds, 1);
  EXPECT_EQ(n2.adds, 1);
  EXPECT_EQ(n3.adds, 1);
  EXPECT_TRUE(n1.inTaggedList());

  // Remove from the middle.
  EXPECT_EQ(ls.remove(n2.getIterator()), &n2);
  EXPECT_EQ(ls.size(), std::distance(ls.begin(), ls.end()));
  EXPECT_EQ(n1.adds, 1);
  EXPECT_EQ(n2.adds, 0);
  EXPECT_EQ(n3.adds, 1);

  // Remove from back.
  EXPECT_EQ(ls.remove(i3), &n3);
  EXPECT_EQ(ls.size(), std::distance(ls.begin(), ls.end()));
  EXPECT_EQ(n1.adds, 1);
  EXPECT_EQ(n2.adds, 0);
  EXPECT_EQ(n3.adds, 0);

  // Remove from front.
  EXPECT_EQ(ls.remove(i1), &n1);
  EXPECT_EQ(ls.size(), std::distance(ls.begin(), ls.end()));
  EXPECT_EQ(n1.adds, 0);
  EXPECT_EQ(n2.adds, 0);
  EXPECT_EQ(n3.adds, 0);
  EXPECT_FALSE(n1.inTaggedList());

  EXPECT_TRUE(ls.empty());
}

TEST(TaggedList, reinsert) {
  List ls;
  Node n1(1);

  EXPECT_FALSE(n1.inTaggedList());
  ls.push_back(&n1);
  EXPECT_TRUE(n1.inTaggedList());
  ls.remove(&n1);
  EXPECT_FALSE(n1.inTaggedList());
  ls.push_back(&n1);
  EXPECT_TRUE(n1.inTaggedList());
  ls.remove(&n1);
}

TEST(TaggedList, reverse_iterator) {
  List ls;
  Node n1(1), n2(2), n3(3);
  ls.push_front(&n1);
  ls.push_back(&n2);
  ls.push_back(&n3);

  EXPECT_EQ(ls.size(), std::distance(ls.rbegin(), ls.rend()));

  auto i = ls.rbegin();
  EXPECT_EQ(i->id, 3);
  ++i;
  EXPECT_EQ(i->id, 2);
  i++;
  EXPECT_EQ(i->id, 1);
  EXPECT_TRUE(++i == ls.rend());

  EXPECT_EQ((--i)->id, 1);
  EXPECT_EQ((i--)->id, 1);
  EXPECT_EQ(i->id, 2);
  --i;
  EXPECT_EQ(i->id, 3);
  EXPECT_TRUE(i == ls.rbegin());

  // Same, with const_reverse_iterator.
  {
    const auto &cls = ls;
    auto i = cls.rbegin();
    EXPECT_EQ(i->id, 3);
    ++i;
    EXPECT_EQ(i->id, 2);
    i++;
    EXPECT_EQ(i->id, 1);
    EXPECT_TRUE(++i == cls.rend());

    EXPECT_EQ((--i)->id, 1);
    EXPECT_EQ((i--)->id, 1);
    EXPECT_EQ(i->id, 2);
    --i;
    EXPECT_EQ(i->id, 3);
    EXPECT_TRUE(i == cls.rbegin());
  }

  ls.clearAndLeakNodesUnsafely();
}

TEST(TaggedList, clearAndLeakNodesUnsafely) {
  List ls;
  Node n1(1), n2(2), n3(3);
  ls.insert(ls.end(), &n1);
  ls.insert(ls.end(), &n2);
  ls.insert(ls.end(), &n3);

  EXPECT_EQ(ls.size(), 3);
  EXPECT_EQ(n1.adds, 1);
  EXPECT_EQ(n2.adds, 1);
  EXPECT_EQ(n3.adds, 1);

  ls.clearAndLeakNodesUnsafely();

  EXPECT_TRUE(ls.begin() == ls.end());
  EXPECT_EQ(ls.size(), 0);
  EXPECT_EQ(n1.adds, 1);
  EXPECT_EQ(n2.adds, 1);
  EXPECT_EQ(n3.adds, 1);
}

TEST(TaggedList, clear) {
  List ls;
  Node *n1 = new Node(1);
  Node *n2 = new Node(2);
  Node *n3 = new Node(3);
  ls.insert(ls.end(), n1);
  ls.insert(ls.end(), n2);
  ls.insert(ls.end(), n3);

  EXPECT_EQ(ls.size(), 3);
  EXPECT_EQ(n1->adds, 1);
  EXPECT_EQ(n2->adds, 1);
  EXPECT_EQ(n3->adds, 1);

  ls.clear();

  EXPECT_TRUE(ls.empty());
  EXPECT_TRUE(ls.begin() == ls.end());
}

TEST(TaggedList, pop) {
  List ls;
  ls.push_front(new Node(1));
  ls.push_back(new Node(2));
  ls.pop_back();
  EXPECT_EQ(ls.front().id, 1);
  EXPECT_EQ(ls.back().id, 1);
  ls.pop_front();
  EXPECT_TRUE(ls.empty());
  EXPECT_TRUE(ls.begin() == ls.end());
}

TEST(TaggedList, inequality) {
  List ls;

  EXPECT_FALSE(ls.begin() < ls.begin());
  EXPECT_TRUE(ls.begin() <= ls.begin());
  EXPECT_FALSE(ls.begin() > ls.begin());
  EXPECT_TRUE(ls.begin() >= ls.begin());

  ls.push_front(new Node(1));
  ls.push_back(new Node(2));

  auto i1 = ls.begin();
  auto i2 = ls.back().getIterator();

  EXPECT_TRUE(i1 < i2);
  EXPECT_TRUE(i1 <= i2);
  EXPECT_FALSE(i1 > i2);
  EXPECT_FALSE(i1 >= i2);
  EXPECT_FALSE(i2 < i1);
  EXPECT_FALSE(i2 <= i1);
  EXPECT_TRUE(i2 > i1);
  EXPECT_TRUE(i2 >= i1);

  EXPECT_TRUE(i1 < ls.end());
  EXPECT_TRUE(i1 <= ls.end());
  EXPECT_FALSE(i1 > ls.end());
  EXPECT_FALSE(i1 >= ls.end());
  EXPECT_FALSE(ls.end() < i1);
  EXPECT_FALSE(ls.end() <= i1);
  EXPECT_TRUE(ls.end() > i1);
  EXPECT_TRUE(ls.end() >= i1);

  // Same thing for reverse iterators.
  auto r1 = ls.rbegin();
  auto r2 = ls.front().getReverseIterator();

  EXPECT_TRUE(r1 < r2);
  EXPECT_TRUE(r1 <= r2);
  EXPECT_FALSE(r1 > r2);
  EXPECT_FALSE(r1 >= r2);
  EXPECT_FALSE(r2 < r1);
  EXPECT_FALSE(r2 <= r1);
  EXPECT_TRUE(r2 > r1);
  EXPECT_TRUE(r2 >= r1);

  EXPECT_TRUE(r1 < ls.rend());
  EXPECT_TRUE(r1 <= ls.rend());
  EXPECT_FALSE(r1 > ls.rend());
  EXPECT_FALSE(r1 >= ls.rend());
  EXPECT_FALSE(ls.rend() < r1);
  EXPECT_FALSE(ls.rend() <= r1);
  EXPECT_TRUE(ls.rend() > r1);
  EXPECT_TRUE(ls.rend() >= r1);
}

TEST(TaggedList, sequencing) {
  List ls;

  for (unsigned i = 0; i < 10000; i++)
    ls.push_back(new Node(i));

  // Verify iterator ordering forwards and backwards.
  for (auto i1 = ls.begin(); i1 != ls.end();) {
    auto i0 = i1++;
    EXPECT_TRUE(i0 < i1);
  }
  for (auto i1 = ls.rbegin(); i1 != ls.rend();) {
    auto i0 = i1++;
    EXPECT_TRUE(i0 < i1);
  }

  Node &middle = ls.front();
  for (unsigned i = 0; i < 10000; i++)
    ls.push_front(new Node(i + 10000));

  for (auto i1 = ls.begin(); i1 != ls.end();) {
    auto i0 = i1++;
    EXPECT_TRUE(i0 < i1);
  }
  for (auto i1 = ls.rbegin(); i1 != ls.rend();) {
    auto i0 = i1++;
    EXPECT_TRUE(i0 < i1);
  }

  for (unsigned i = 0; i < 10000; i++)
    ls.insert(middle.getIterator(), new Node(i + 20000));

  for (auto i1 = ls.begin(); i1 != ls.end();) {
    auto i0 = i1++;
    EXPECT_TRUE(i0 < i1);
  }
  for (auto i1 = ls.rbegin(); i1 != ls.rend();) {
    auto i0 = i1++;
    EXPECT_TRUE(i0 < i1);
  }
}
