/**
 * Copyright (c) 2017-present, Facebook, Inc.
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

#include "../../lib/IR/GraphScheduler.h"

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/PlaceholderBindings.h"

#include "gtest/gtest.h"

using namespace glow;

/// Tests a case in which the memory required to store a node's
/// output is greater than the memory required to store its input.
/// This node uses more memory after it executes, and should be
/// scheduled after its siblings that free up memory after
/// they execute.
TEST(GraphScheduler, testMaxSizeLessThanResultSize) {
  Module MD;
  PlaceholderBindings bindings;
  auto *smallTensorA =
      MD.createPlaceholder(ElemKind::FloatTy, {1, 4, 4}, "small_1", false);
  bindings.allocate(smallTensorA);
  auto *smallTensorB =
      MD.createPlaceholder(ElemKind::FloatTy, {1, 4, 4}, "small_2", false);
  bindings.allocate(smallTensorB);
  auto *bigTensor =
      MD.createPlaceholder(ElemKind::FloatTy, {100, 4, 4}, "big", false);
  bindings.allocate(bigTensor);
  Function *F = MD.createFunction("F");
  Node *transposeBig = F->createTranspose("transposeBig", bigTensor, {0, 2, 1});
  Node *sliceBig =
      F->createSlice("sliceBig", transposeBig, {0, 0, 0}, {1, 4, 4});
  Node *concatSmall =
      F->createConcat("concatSmall", {smallTensorA, smallTensorB}, 0);
  F->createConcat("concat", {concatSmall, sliceBig}, 0);

  // The graph created above looks like this:
  //
  //  bigTensor       smallTensorA       smallTensorB
  // {100, 4, 4}        {1, 4, 4}         {1, 4, 4}
  //     |                     \         /
  //     v                      v       v
  // transposeBig {0, 2, 1}    concatSmall {0}
  //    {100, 4, 4}              {2, 4, 4}
  //     |                           |
  //     v                           |
  //  sliceBig {0, 0, 0}, {1, 4, 4}  |
  //    {1, 4, 4}                    |
  //     |                           |
  //     |                           |
  //     |                           |
  //     --------> concat {0} <-------
  //               {3, 4, 4}
  //

  {
    // Since all of the tensors are Variables, they don't need
    // memory for storing their outputs. Consequently, sliceBig
    // should be scheduled before concatSmall in this example
    // because the former frees up some memory while the latter
    // uses up more memory after execution.
    NodesPtrList schedule;
    ChildMemSizeBasedScheduler scheduler(*F, schedule);
    scheduler.schedule();

    // Find the positions of sliceBig and concatSmall in
    // the schedule.
    auto concatSmallIt =
        std::find(schedule.begin(), schedule.end(), concatSmall);
    auto sliceBigIt = std::find(schedule.begin(), schedule.end(), sliceBig);

    // For the reason given above, sliceBig should be scheduled
    // before concatSmall.
    EXPECT_LT(std::distance(schedule.begin(), sliceBigIt),
              std::distance(schedule.begin(), concatSmallIt));
  }

  {
    // The graph will be traversed in post order. The root
    // node is concat node in this case. Then, concatSmall node
    // will be visited, since it's the left operand of concat node.
    // Consequently, sliceBig should be scheduled after concatSmall.

    NodesPtrList schedule;
    TopologicalSortBasedScheduler scheduler(*F, schedule);
    scheduler.schedule();

    // Find the positions of sliceBig and concatSmall in
    // the schedule.
    auto concatSmallIt =
        std::find(schedule.begin(), schedule.end(), concatSmall);
    auto sliceBigIt = std::find(schedule.begin(), schedule.end(), sliceBig);

    // For the reason given above, sliceBig should be scheduled
    // after concatSmall.
    EXPECT_GT(std::distance(schedule.begin(), sliceBigIt),
              std::distance(schedule.begin(), concatSmallIt));
  }
}

TEST(GraphScheduler, ScheduleQuantizationProfileRightAfterNodeBeingProfiled) {
  Module MD;
  PlaceholderBindings bindings;
  auto *input1 =
      MD.createPlaceholder(ElemKind::FloatTy, {1, 4, 4}, "input1", false);
  bindings.allocate(input1);
  auto *input2 =
      MD.createPlaceholder(ElemKind::FloatTy, {1, 4, 4}, "input2", false);
  bindings.allocate(input2);
  Function *F = MD.createFunction("F");
  Node *add = F->createAdd("add", input1, input2);
  Node *sub = F->createSub("sub", input1, input2);
  Node *mul = F->createMul("mul", add, sub);
  Node *save = F->createSave("save", mul);
  Node *quantizationProfileAdd =
      F->createQuantizationProfile(bindings, "qpAdd", add);
  Node *quantizationProfileSub =
      F->createQuantizationProfile(bindings, "qpSub", sub);

  // Since all of the tensors are Variables, they don't need
  // memory for storing their outputs. Consequently, sliceBig
  // should be scheduled before concatSmall in this example
  // because the former frees up some memory while the latter
  // uses up more memory after execution.
  NodesPtrList schedule;
  ChildMemSizeBasedScheduler scheduler(*F, schedule);
  scheduler.schedule();

  // Find the positions of add and quantizationProfileAdd in the schedule.
  auto addIt = std::find(schedule.begin(), schedule.end(), add);
  auto qpAddIt =
      std::find(schedule.begin(), schedule.end(), quantizationProfileAdd);
  // Expect the quantization profiling node to be scheduled right after the node
  // being profiled.
  EXPECT_EQ(++addIt, qpAddIt);

  // Find the positions of sub and quantizationProfileSub in the schedule.
  auto subIt = std::find(schedule.begin(), schedule.end(), sub);
  auto qpSubIt =
      std::find(schedule.begin(), schedule.end(), quantizationProfileSub);
  // Expect the quantization profiling node to be scheduled right after the node
  // being profiled.
  EXPECT_EQ(++subIt, qpSubIt);

  // Expect the save node to be the last in the schedule.
  EXPECT_EQ(save, schedule.back());
}
