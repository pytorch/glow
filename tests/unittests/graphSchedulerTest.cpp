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

#include "glow/IR/GraphScheduler.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"

#include "gtest/gtest.h"

using namespace glow;

/// Tests a case in which the memory required to store a node's
/// output is greater than the memory required to store its input.
/// This node uses more memory after it executes, and should be
/// scheduled after its siblings that free up memory after
/// they execute.
TEST(GraphScheduler, testMaxSizeLessThanResultSize) {
  Module MD;
  Variable *smallTensorA = MD.createVariable(ElemKind::FloatTy, {1, 4, 4},
                                             "small_1", VisibilityKind::Public);
  Variable *smallTensorB = MD.createVariable(ElemKind::FloatTy, {1, 4, 4},
                                             "small_2", VisibilityKind::Public);
  Variable *bigTensor = MD.createVariable(ElemKind::FloatTy, {100, 4, 4}, "big",
                                          VisibilityKind::Public);

  Function *F = MD.createFunction("F");
  Node *transposeBig = F->createTranspose("transposeBig", bigTensor, {0, 2, 1});
  Node *sliceBig =
      F->createSlice("sliceBig", transposeBig, {0, 0, 0}, {1, 4, 4});
  Node *concatSmall =
      F->createConcat("concatSmall", {smallTensorA, smallTensorB}, 0);
  F->createConcat("concat", {concatSmall, sliceBig}, 0);

  /// The graph created above looks like this:
  ///
  ///  bigTensor       smallTensorA       smallTensorB
  /// {100, 4, 4}        {1, 4, 4}         {1, 4, 4}
  ///     |                     \         /
  ///     v                      v       v
  /// transposeBig                 concatSmall
  ///  {0, 2, 1}                     {0}
  ///     |                           |
  ///     v                           |
  ///  sliceBig                       |
  ///  {0, 0, 0}                      |
  ///  {1, 4, 4}                      |
  ///     |                           |
  ///     ---------> concat <----------
  ///                 {0}
  ///
  /// Since all of the tensors are Variables, they don't need
  /// memory for storing their outputs. Consequently, sliceBig
  /// should be scheduled before concatSmall in this example
  /// because the former frees up some memory while the latter
  /// uses up more memory after execution.
  NodesPtrList schedule;
  ChildMemSizeBasedScheduler scheduler(*F, schedule);
  scheduler.schedule();

  /// Find the positions of sliceBig and concatSmall in
  /// the schedule.
  int i = 0, concatSmallIdx = -1, sliceBigIdx = -1;
  for (auto *N : schedule) {
    if (N == concatSmall) {
      concatSmallIdx = i;
    }

    if (N == sliceBig) {
      sliceBigIdx = i;
    }

    ++i;
  }

  /// For the reason given above, sliceBig should be scheduled
  /// before concatSmallIdx.
  ASSERT_LT(sliceBigIdx, concatSmallIdx);
}
